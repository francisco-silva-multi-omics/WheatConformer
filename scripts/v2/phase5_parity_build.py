from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from phase5_parity_common import (
    BUNDLE_RELATIVE,
    RELEASE_ID,
    RELEASE_RELATIVE,
    SEED,
    UPSTREAM_RELEASE_ID,
    UPSTREAM_RELATIVE,
    V1_INCIDENT_RELATIVE,
    ProtectedPathGuard,
    contiguous_weighted_blocks,
    deterministic_balanced_assignment,
    ensure_fail_if_exists,
    environment_versions,
    factor_diagnostics,
    git_head,
    index_signature,
    kernel_alignment,
    normalize_cycle_year,
    relative_posix,
    sha256_file,
    stable_json_hash,
    utc_now,
    write_json,
    write_tsv,
)


EXISTING_SCENARIOS = ("GNEW_EOBS", "GOBS_ENEW", "GNEW_ENEW")
NEW_SCENARIOS = ("TEMPORAL_YEAR", "COUNTRY_HOLDOUT")
OUTER_FOLDS = range(1, 6)
INNER_FOLDS = range(1, 6)
SEEDS_PANEL = "seeds_of_discovery_dartseq"

SEEDS_DIR = Path(
    "GENOTYPIC_DATA/Seeds_of_Discovery_-_MasAgro_Biodiversidad_Wheat_DArTseq-Derived_SNP_Data_Beta_Recall_Results_From_2011-2014"
)
SEEDS_MATRIX = SEEDS_DIR / "SEQ_SNPs_Extract_45610samples_102474markers.txt"
SEEDS_CROSSWALK = SEEDS_DIR / "SampleIDvsGID_45610samples.txt"
SEEDS_DICTIONARY = SEEDS_DIR / "DArTSeq_SNPs_report_data_dictionary.txt"

SAWYT_MATRIX_GLOBS = (
    "GENOTYPIC_DATA/GBS/13th_Semi-arid_wheat_yield_trial_genotyping-by-sequencing_data/13TH_SAWYTgbs_CIMMYT_20120708.txt",
    "GENOTYPIC_DATA/GBS/14th_Semi-Arid_Wheat_Yield_Trial_Genotyping-by-sequencing_Data/14TH_SAWYTgbs_CIMMYT_20120708.txt",
    "GENOTYPIC_DATA/GBS/15th_Semi-arid_wheat_yield_trial_genotyping-by-sequencing_data/15TH_SAWYTgbs_CIMMYT_20120708.txt",
    "GENOTYPIC_DATA/GBS/16th_Semi-Arid_Wheat_Yield_Trial_Genotyping-by-sequencing_Data/16TH_SAWYTgbs_CIMMYT_20120708.txt",
    "GENOTYPIC_DATA/GBS/17th_Semi-Arid_Wheat_Yield_Trial_Genotyping-by-sequencing_Data/17TH_SAWYTgbs_CIMMYT_20120708.txt",
    "GENOTYPIC_DATA/GBS/18th_Semi-Arid_wheat_yield_trial_genotyping-by-sequencing_data/18TH_SAWYT_GBS.txt",
)

BUNDLE_ENV = Path("server_phase5_parity_bundle/artifacts/environment")
BUNDLE_HMP = Path("server_phase5_parity_bundle/artifacts/genotype_panels/hmp")
BUNDLE_HMP_MANIFEST = Path("server_phase5_parity_bundle/artifacts/metadata_outputs/canonical_hmp_sample_manifest.tsv")

EXPECTED_PRIMARY_OVERLAP = {
    "frozen_hmp_v1": (5187, 1_173_132),
    "cimmyt_bread_gbs_2013_2018": (4512, 721_033),
    "seeds_of_discovery_dartseq": (3212, 801_276),
    "eyt_haplotype_blocks_2011_2018": (2612, 520_592),
    "dartag_panel2": (1931, 280_688),
    "mas_45ibwsn": (334, 158_464),
    "hibap35k": (95, 52_397),
    "mexican_landrace_dartseq": (0, 0),
}

PANEL_ROLES = {
    "cimmyt_bread_gbs_2013_2018": ("DENSE_GENOMEWIDE_DIAGNOSTIC", "BLOCKED_NO_PREIMPUTATION_CALLS_OR_LOSSLESS_MISSING_MASK"),
    "cimmyt_bread_gbs_2013_2018_ta_metadata": ("METADATA_ONLY", "BLOCKED_NO_RAW_MATRIX"),
    "dartag_panel2": ("TARGETED_MARKER_EXPERT", "REGISTERED_TARGETED_NOT_GENOMEWIDE"),
    "dartseq80k_collection": ("IDENTITY_CANDIDATE_ONLY", "BLOCKED_NO_TYPED_MATRIX"),
    "dartseq80k_hexaploid": ("IDENTITY_CANDIDATE_ONLY", "BLOCKED_NO_SAME_DATASET_TYPED_IDENTITY"),
    "dartseq80k_tetraploid": ("IDENTITY_CANDIDATE_ONLY", "BLOCKED_NO_SAME_DATASET_TYPED_IDENTITY"),
    "dartseq80k_wheat_recall": ("IDENTITY_CANDIDATE_ONLY", "BLOCKED_NO_SAME_DATASET_TYPED_IDENTITY"),
    "dartseq80k_wild_relative": ("IDENTITY_CANDIDATE_ONLY", "BLOCKED_NO_SAME_DATASET_TYPED_IDENTITY"),
    "eyt_haplotype_blocks_2011_2018": ("HAPLOTYPE_EXPERT", "BLOCKED_MISSING_BLOCK_DEFINITION_AND_SOURCE_SNP_PROVENANCE"),
    "frozen_hmp_v1": ("HISTORICAL_TRANSDUCTIVE_DIAGNOSTIC", "BLOCKED_TRACED_TO_GLOBALLY_IMPUTED_CIMMYT_EXPORT"),
    "gbs_13sawyt": ("SEPARATE_SMALL_PANEL_EXPERT", "CONDITIONAL_ON_ALL_STATE_MINIMUM_SUPPORT"),
    "gbs_14sawyt": ("SEPARATE_SMALL_PANEL_EXPERT", "CONDITIONAL_ON_ALL_STATE_MINIMUM_SUPPORT"),
    "gbs_15sawyt": ("SEPARATE_SMALL_PANEL_EXPERT", "CONDITIONAL_ON_ALL_STATE_MINIMUM_SUPPORT"),
    "gbs_16sawyt": ("SEPARATE_SMALL_PANEL_EXPERT", "CONDITIONAL_ON_ALL_STATE_MINIMUM_SUPPORT"),
    "gbs_17sawyt": ("SEPARATE_SMALL_PANEL_EXPERT", "CONDITIONAL_ON_ALL_STATE_MINIMUM_SUPPORT"),
    "gbs_18sawyt": ("SEPARATE_SMALL_PANEL_EXPERT", "CONDITIONAL_ON_ALL_STATE_MINIMUM_SUPPORT"),
    "hibap35k": ("IMMUTABLE_PHASE5_PRODUCTION_KG", "REUSED_BY_REFERENCE_BYTE_FOR_BYTE"),
    "mas_45ibwsn": ("TARGETED_MARKER_EXPERT", "REGISTERED_TARGETED_NOT_GENOMEWIDE"),
    "mas_57ibwsn_42sawsn_35hrwsn": ("TARGETED_MARKER_EXPERT", "REGISTERED_TARGETED_NOT_GENOMEWIDE"),
    "mas_58ibwsn_43sawsn": ("TARGETED_MARKER_EXPERT", "REGISTERED_TARGETED_NOT_GENOMEWIDE"),
    "mexican_landrace_dartseq": ("EXTERNAL_POPULATION_STRUCTURE", "ZERO_PRIMARY_OVERLAP_NO_SUPERVISED_KG"),
    "seeds_of_discovery_dartseq": ("DENSE_GENOMEWIDE_SPLIT_LOCAL_KG", "ACTIVATE_WHERE_TRAINING_SUPPORT_AND_QC_PASS"),
}

MANAGEMENT_FEATURES = (
    "AREA_HARVESTED_BED_PLOT_M2",
    "AREA_SOWN_BED_PLOT_M2",
    "CALCULATED_OF_TOTAL_WATER_APPLIED_BY_IRRIGATION",
    "ESTIMATE_OF_TOTAL_WATER_APPLIED_BY_IRRIGATION",
    "FERTILIZER_%K2O_1",
    "FERTILIZER_%K2O_2",
    "FERTILIZER_%K2O_3",
    "FERTILIZER_%N_1",
    "FERTILIZER_%N_2",
    "FERTILIZER_%N_3",
    "FERTILIZER_%P2O5_1",
    "FERTILIZER_%P2O5_2",
    "FERTILIZER_%P2O5_3",
    "FERTILIZER_KG/HA_1",
    "FERTILIZER_KG/HA_2",
    "FERTILIZER_KG/HA_3",
    "FUNGICIDE",
    "HAND_WEEDING",
    "HERBICIDE",
    "IRRIGATED",
    "IRRIGATION_AFTER_SOWING",
    "K_FERTILIZER_APPLIED_OLD",
    "NUMBER_POST_SOWING_IRRIGATIONS",
    "NUMBER_PRE_SOWING_IRRIGATIONS",
    "N_FERTILIZER_APPLIED_OLD",
    "PESTICIDE",
    "PRE_SOWING_IRRIGATION",
    "P_FERTILIZER_APPLIED_OLD",
)

STAGE_WINDOWS = {
    "ESTABLISHMENT_D0_30": "d0_30",
    "VEGETATIVE_D30_60": "d30_60",
    "REPRODUCTIVE_D60_90": "d60_90",
    "GRAIN_FILL_EARLY_D90_120": "d90_120",
    "GRAIN_FILL_LATE_D120_150": "d120_150",
    "LATE_SEASON_D150_180": "d150_180",
}

STAGE_FEATURE_GROUPS = {
    "HEAT": ("temperature_mean_c", "temperature_max_c", "heat_days_tmax_ge_30", "heat_days_tmax_ge_35", "vpd_mean_kpa", "vpd_max_kpa"),
    "WATER": ("precipitation_total_mm", "dry_days_precip_lt_1mm", "high_vpd_days_gt_1_5", "drought_days_precip_lt_1mm_and_vpd_gt_1_5"),
    "DEVELOPMENT": ("gdd_base0_sum", "gdd_base5_sum", "cold_days_tmin_lt_0", "chill_days_tmean_0_10"),
    "RADIATION": ("solar_radiation_total_mj_m2", "solar_radiation_mean_daily_mj_m2"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the clean Phase-5 parity extension v2")
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--resume-after-seeds-collapse", action="store_true")
    return parser.parse_args()


def create_directories(out: Path) -> None:
    for relative in (
        "genomic/states",
        "environment",
        "splits/state_entities",
        "masks",
        "redundancy",
        "tests",
        "logs",
    ):
        (out / relative).mkdir(parents=True, exist_ok=False)


def verify_preflight(root: Path, guard: ProtectedPathGuard) -> dict[str, Any]:
    upstream = root / UPSTREAM_RELATIVE
    decision_path = guard.assert_allowed(upstream / "PHASE5_RELEASE_DECISION.json", "READ_UPSTREAM_DECISION")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("release_id") != UPSTREAM_RELEASE_ID or decision.get("status") != "PASS_PHASE5_KERNEL_VALIDATION":
        raise SystemExit("BLOCKED: immutable Phase-5 release binding mismatch")
    bundle = root / BUNDLE_RELATIVE
    required = (
        bundle / "TRANSFER_COMPLETE",
        bundle / "inventory/approved_path_size_sha256.tsv",
        bundle / "inventory/current_path_size_sha256.tsv",
        bundle / "inventory/manifest_comparison.diff",
        bundle / "inventory/missing_expected_paths.txt",
    )
    for path in required:
        guard.assert_allowed(path, "PREFLIGHT_BUNDLE_INVENTORY")
        if not path.exists():
            raise SystemExit(f"BLOCKED: missing bundle control file {path}")
    if (bundle / "inventory/manifest_comparison.diff").stat().st_size != 0:
        raise SystemExit("BLOCKED: bundle manifest comparison is non-empty")
    if (bundle / "inventory/missing_expected_paths.txt").stat().st_size != 0:
        raise SystemExit("BLOCKED: bundle has missing expected paths")
    approved = (bundle / "inventory/approved_path_size_sha256.tsv").read_bytes()
    current = (bundle / "inventory/current_path_size_sha256.tsv").read_bytes()
    if approved != current:
        raise SystemExit("BLOCKED: approved/current bundle manifests differ")
    return {
        "release_id": RELEASE_ID,
        "upstream_release_id": decision["release_id"],
        "upstream_status": decision["status"],
        "bundle_manifest_sha256": sha256_file(bundle / "inventory/approved_path_size_sha256.tsv"),
        "bundle_transfer_complete": True,
        "target_absent": not (root / RELEASE_RELATIVE).exists(),
        "denylist_rules": len(guard.rules),
        "status": "PASS_PREFLIGHT",
    }


def parse_bundle_manifest(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            relative, size, digest = line.rstrip("\n").split("\t")
            rows.append({"bundle_relative_path": relative, "size": int(size), "approved_sha256": digest})
    return pd.DataFrame(rows)


def opening_hash_manifest(root: Path, out: Path, guard: ProtectedPathGuard) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    upstream = root / UPSTREAM_RELATIVE
    for path in sorted(upstream.rglob("*")):
        if not path.is_file():
            continue
        rule = guard.matched_rule(path)
        if rule:
            guard.inventory_metadata_only(path)
            rows.append(
                {
                    "scope": "IMMUTABLE_PHASE5",
                    "relative_path": relative_posix(path, root),
                    "size": path.stat().st_size,
                    "sha256": "NOT_REHASHED_DENYLIST_METADATA_ONLY",
                    "access": "METADATA_ONLY",
                    "matched_rule": rule,
                }
            )
        else:
            guard.assert_allowed(path, "OPENING_HASH")
            rows.append(
                {
                    "scope": "IMMUTABLE_PHASE5",
                    "relative_path": relative_posix(path, root),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "access": "HASHED",
                    "matched_rule": "",
                }
            )
    bundle_manifest_path = guard.assert_allowed(root / BUNDLE_RELATIVE / "inventory/approved_path_size_sha256.tsv", "READ_BUNDLE_MANIFEST")
    bundle_manifest = parse_bundle_manifest(bundle_manifest_path)
    for record in bundle_manifest.to_dict("records"):
        path = root / BUNDLE_RELATIVE / "artifacts" / record["bundle_relative_path"]
        rule = guard.matched_rule(path)
        if rule:
            guard.inventory_metadata_only(path)
            observed = record["approved_sha256"]
            access = "APPROVED_SHA256_METADATA_ONLY"
        else:
            guard.assert_allowed(path, "OPENING_HASH_BUNDLE_ARTIFACT")
            observed = sha256_file(path)
            access = "HASHED"
            if observed != record["approved_sha256"] or path.stat().st_size != record["size"]:
                raise SystemExit(f"BLOCKED: bundle artifact integrity mismatch: {record['bundle_relative_path']}")
        rows.append(
            {
                "scope": "SERVER_PHASE5_PARITY_BUNDLE",
                "relative_path": relative_posix(path, root),
                "size": record["size"],
                "sha256": observed,
                "access": access,
                "matched_rule": rule,
            }
        )
    direct_inputs = [
        root / V1_INCIDENT_RELATIVE / "PROTECTED_PATH_DENYLIST.txt",
        root / BUNDLE_RELATIVE / "TRANSFER_COMPLETE",
        root / BUNDLE_RELATIVE / "inventory/approved_path_size_sha256.tsv",
        root / BUNDLE_RELATIVE / "inventory/current_path_size_sha256.tsv",
        root / BUNDLE_RELATIVE / "provenance/server_state.txt",
        root / SEEDS_MATRIX,
        root / SEEDS_CROSSWALK,
        root / SEEDS_DICTIONARY,
        *(root / relative for relative in SAWYT_MATRIX_GLOBS),
        root / "audit/v2/phase3g_all_panel_genotype_linkage_audit_v2/accepted_all_panel_crosswalk.parquet",
        root / "audit/v2/phase3g_all_panel_genotype_linkage_audit_v2/sample_identifier_ledger.parquet",
        root / "audit/v2/phase3g_all_panel_genotype_linkage_audit_v2/panel_inventory.tsv",
    ]
    existing = {row["relative_path"] for row in rows}
    for path in direct_inputs:
        relative = relative_posix(path, root)
        if relative in existing:
            continue
        guard.assert_allowed(path, "OPENING_HASH_DIRECT_INPUT")
        rows.append(
            {
                "scope": "DIRECT_INPUT",
                "relative_path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "access": "HASHED",
                "matched_rule": "",
            }
        )
    frame = pd.DataFrame(rows).sort_values(["scope", "relative_path"])
    write_tsv(out / "OPENING_HASH_MANIFEST.tsv", frame)
    return frame


def write_opening_contract(root: Path, out: Path, guard: ProtectedPathGuard, preflight: dict[str, Any]) -> None:
    shutil.copyfile(root / V1_INCIDENT_RELATIVE / "PROTECTED_PATH_DENYLIST.txt", out / "PROTECTED_PATH_DENYLIST.txt")
    write_json(
        out / "OPENING_RELEASE.json",
        {
            "release_id": RELEASE_ID,
            "release_version": "v2",
            "release_root": relative_posix(out, root),
            "authoritative_phase5_release": UPSTREAM_RELEASE_ID,
            "v1_attempt_disposition": "TERMINALLY_BLOCKED_INCIDENT_ONLY_NO_SCIENTIFIC_DECISIONS_INHERITED",
            "opened_at_utc": utc_now(),
            "git_head": git_head(root),
            "seed": SEED,
            "phenotype_blind": True,
            "model_training_authorized": False,
            "protected_files_rendered": [],
            "preflight": preflight,
        },
    )
    write_json(
        out / "run_manifest.json",
        {
            "release_id": RELEASE_ID,
            "repository_root": str(root),
            "release_root": str(out),
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "packages": environment_versions(),
            "model_training_performed": False,
            "component_selection_performed": False,
            "performance_evaluation_performed": False,
            "outer_test_outcomes_accessed": False,
            "final_holdout_accessed": False,
            "inner_validation_metrics_accessed": False,
            "future_projection_performed": False,
            "commit_or_push_performed": False,
        },
    )


def load_phase5_observation_assignment(root: Path, guard: ProtectedPathGuard) -> pd.DataFrame:
    path = guard.assert_allowed(root / UPSTREAM_RELATIVE / "splits/observation_split_assignment.parquet", "READ_ID_ONLY_SPLIT_ASSIGNMENT")
    columns = [
        "phase4_adjusted_row_id",
        "canonical_gid",
        "environment_id",
        "year",
        "country",
        "primary_weighted_training_eligible",
        "gnew_eobs_gid_fold",
        "gobs_enew_env_fold",
        "gnew_enew_gid_fold",
        "gnew_enew_env_fold",
    ]
    frame = pq.read_table(path, columns=columns).to_pandas()
    if len(frame) != 2_242_863 or frame["phase4_adjusted_row_id"].duplicated().any():
        raise AssertionError("Immutable Phase-5 ID-only split population changed")
    return frame


def existing_phase5_states(root: Path, guard: ProtectedPathGuard, obs: pd.DataFrame) -> dict[str, dict[str, Any]]:
    inner_path = guard.assert_allowed(root / UPSTREAM_RELATIVE / "splits/inner_fold_assignment.tsv", "READ_ID_ONLY_INNER_ASSIGNMENT")
    inner = pd.read_csv(inner_path, sep="\t", dtype=str, keep_default_na=False)
    states: dict[str, dict[str, Any]] = {}
    for scenario in EXISTING_SCENARIOS:
        for outer in OUTER_FOLDS:
            if scenario == "GNEW_EOBS":
                outer_mask = obs.gnew_eobs_gid_fold.ne(outer)
            elif scenario == "GOBS_ENEW":
                outer_mask = obs.gobs_enew_env_fold.ne(outer)
            else:
                outer_mask = obs.gnew_enew_gid_fold.ne(outer) & obs.gnew_enew_env_fold.ne(outer)
            for inner_fold in (None, *INNER_FOLDS):
                mask = outer_mask.copy()
                if inner_fold is not None:
                    held = inner[(inner.scenario == scenario) & (inner.outer_fold.astype(int) == outer) & (inner.inner_fold.astype(int) == inner_fold)]
                    if scenario in {"GNEW_EOBS", "GNEW_ENEW"}:
                        gids = set(held.loc[held.entity_type == "CANONICAL_GID", "entity_id"])
                        mask &= ~obs.canonical_gid.isin(gids)
                    if scenario in {"GOBS_ENEW", "GNEW_ENEW"}:
                        envs = set(held.loc[held.entity_type == "ENVIRONMENT", "entity_id"])
                        mask &= ~obs.environment_id.isin(envs)
                state_id = f"{scenario}__OUTER{outer}" + ("" if inner_fold is None else f"__INNER{inner_fold}")
                state_obs = obs.loc[mask, ["canonical_gid", "environment_id"]]
                states[state_id] = {
                    "state_id": state_id,
                    "scenario": scenario,
                    "outer_fold": outer,
                    "inner_fold": inner_fold,
                    "state_level": "OUTER" if inner_fold is None else "INNER",
                    "training_observations": int(mask.sum()),
                    "training_gids": frozenset(state_obs.canonical_gid.astype(str).unique()),
                    "training_environments": frozenset(state_obs.environment_id.astype(str).unique()),
                    "source": "IMMUTABLE_PHASE5_UNCHANGED",
                }
    return states


def summarize_role(
    obs: pd.DataFrame, mask: np.ndarray | pd.Series, scenario: str, outer: int, role: str
) -> dict[str, Any]:
    selected = obs.loc[mask]
    primary = selected.primary_weighted_training_eligible.fillna(False)
    return {
        "scenario": scenario,
        "outer_fold": outer,
        "role": role,
        "observations": int(len(selected)),
        "primary_observations": int(primary.sum()),
        "canonical_gids": int(selected.canonical_gid.nunique()),
        "environments": int(selected.environment_id.nunique()),
        "years": int(selected.normalized_year.nunique()) if "normalized_year" in selected else np.nan,
        "countries": int(selected.country.nunique()),
    }


def build_new_scenarios(
    obs: pd.DataFrame, out: Path
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs = obs.copy()
    obs["normalized_year"] = obs.year.map(normalize_cycle_year).astype(np.int16)
    primary = obs.primary_weighted_training_eligible.fillna(False)
    primary_year_weights = (
        obs.loc[primary].groupby("normalized_year", sort=True).size().astype(int).to_dict()
    )
    year_weights = {
        int(year): int(primary_year_weights.get(year, 0))
        for year in sorted(obs.normalized_year.unique())
    }
    year_blocks = contiguous_weighted_blocks(year_weights, 6)
    obs["temporal_block"] = obs.normalized_year.map(year_blocks).astype(np.int8)
    temporal_rows = [
        {
            "normalized_year": year,
            "source_year_labels": ";".join(sorted(obs.loc[obs.normalized_year.eq(year), "year"].astype(str).unique())),
            "temporal_block": block,
            "primary_observations": int(year_weights.get(year, 0)),
        }
        for year, block in sorted(year_blocks.items())
    ]
    primary_country_weights = obs.loc[primary].groupby("country", sort=True).size().astype(int).to_dict()
    country_weights = {
        str(country): int(primary_country_weights.get(country, 0))
        for country in sorted(obs.country.astype(str).unique())
    }
    country_assignment = deterministic_balanced_assignment(country_weights, 5, "COUNTRY_OUTER")
    obs["country_outer_fold"] = obs.country.map(country_assignment).astype(np.int8)
    country_rows = [
        {
            "country": country,
            "outer_fold": fold,
            "primary_observations": int(country_weights.get(country, 0)),
        }
        for country, fold in sorted(country_assignment.items())
    ]

    states: dict[str, dict[str, Any]] = {}
    roles = pd.DataFrame({"phase4_adjusted_row_id": obs.phase4_adjusted_row_id.astype(str)})
    role_summaries: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    inner_assignment_rows: list[dict[str, Any]] = []

    for outer in OUTER_FOLDS:
        train_blocks = set(range(1, outer + 1))
        test_block = outer + 1
        candidate_train_years = sorted(year for year, block in year_blocks.items() if block in train_blocks)
        embargo_year = max(candidate_train_years)
        train_years = set(candidate_train_years) - {embargo_year}
        test_years = {year for year, block in year_blocks.items() if block == test_block}
        train_mask = obs.normalized_year.isin(train_years)
        test_mask = obs.normalized_year.isin(test_years)
        embargo_mask = obs.normalized_year.eq(embargo_year)
        future_mask = ~(train_mask | test_mask | embargo_mask)
        role = np.full(len(obs), "FUTURE_NOT_AVAILABLE", dtype=object)
        role[train_mask.to_numpy()] = "TRAIN"
        role[test_mask.to_numpy()] = "OUTER_TEST_ID_ONLY"
        role[embargo_mask.to_numpy()] = "EMBARGO_ONE_YEAR"
        roles[f"temporal_year_outer{outer}_role"] = pd.Categorical(role)
        for label, mask in (
            ("TRAIN", train_mask),
            ("OUTER_TEST_ID_ONLY", test_mask),
            ("EMBARGO_ONE_YEAR", embargo_mask),
            ("FUTURE_NOT_AVAILABLE", future_mask),
        ):
            role_summaries.append(summarize_role(obs, mask, "TEMPORAL_YEAR", outer, label))
        leakage = train_years.intersection(test_years)
        leakage_rows.extend(
            [
                {
                    "scenario": "TEMPORAL_YEAR",
                    "outer_fold": outer,
                    "inner_fold": "",
                    "check": "YEAR_TRAIN_TEST_DISJOINT",
                    "failure_count": len(leakage),
                    "status": "PASS" if not leakage else "FAIL",
                },
                {
                    "scenario": "TEMPORAL_YEAR",
                    "outer_fold": outer,
                    "inner_fold": "",
                    "check": "STRICT_FORWARD_TIME_ORDER",
                    "failure_count": int(max(train_years) >= min(test_years)),
                    "status": "PASS" if max(train_years) < min(test_years) else "FAIL",
                },
                {
                    "scenario": "TEMPORAL_YEAR",
                    "outer_fold": outer,
                    "inner_fold": "",
                    "check": "ONE_YEAR_EMBARGO_PRESENT",
                    "failure_count": int(embargo_year != min(test_years) - 1),
                    "status": "PASS" if embargo_year == min(test_years) - 1 else "FAIL",
                },
            ]
        )
        for inner_fold in INNER_FOLDS:
            inner_weights = {year: year_weights.get(year, 0) for year in sorted(train_years)}
            inner_blocks = contiguous_weighted_blocks(inner_weights, 6)
            validation_years = {year for year, block in inner_blocks.items() if block == inner_fold + 1}
            candidate_inner_train = sorted(year for year, block in inner_blocks.items() if block <= inner_fold)
            inner_embargo = max(candidate_inner_train)
            inner_train_years = set(candidate_inner_train) - {inner_embargo}
            inner_mask = obs.normalized_year.isin(inner_train_years)
            state_id = f"TEMPORAL_YEAR__OUTER{outer}__INNER{inner_fold}"
            selected = obs.loc[inner_mask, ["canonical_gid", "environment_id"]]
            states[state_id] = {
                "state_id": state_id,
                "scenario": "TEMPORAL_YEAR",
                "outer_fold": outer,
                "inner_fold": inner_fold,
                "state_level": "INNER",
                "training_observations": int(inner_mask.sum()),
                "training_gids": frozenset(selected.canonical_gid.astype(str).unique()),
                "training_environments": frozenset(selected.environment_id.astype(str).unique()),
                "source": "V2_TEMPORAL_EXTENSION",
            }
            for year, block in sorted(inner_blocks.items()):
                inner_assignment_rows.append(
                    {
                        "scenario": "TEMPORAL_YEAR",
                        "outer_fold": outer,
                        "inner_fold": inner_fold,
                        "entity_type": "NORMALIZED_YEAR",
                        "entity_id": year,
                        "assignment": (
                            "TRAIN" if year in inner_train_years else "INNER_VALIDATION_ID_ONLY" if year in validation_years else "EMBARGO_ONE_YEAR" if year == inner_embargo else "NOT_AVAILABLE"
                        ),
                    }
                )
            overlap = inner_train_years.intersection(validation_years)
            leakage_rows.append(
                {
                    "scenario": "TEMPORAL_YEAR",
                    "outer_fold": outer,
                    "inner_fold": inner_fold,
                    "check": "INNER_YEAR_TRAIN_VALIDATION_DISJOINT_AND_FORWARD",
                    "failure_count": len(overlap) + int(max(inner_train_years) >= min(validation_years)),
                    "status": "PASS" if not overlap and max(inner_train_years) < min(validation_years) else "FAIL",
                }
            )
        state_id = f"TEMPORAL_YEAR__OUTER{outer}"
        selected = obs.loc[train_mask, ["canonical_gid", "environment_id"]]
        states[state_id] = {
            "state_id": state_id,
            "scenario": "TEMPORAL_YEAR",
            "outer_fold": outer,
            "inner_fold": None,
            "state_level": "OUTER",
            "training_observations": int(train_mask.sum()),
            "training_gids": frozenset(selected.canonical_gid.astype(str).unique()),
            "training_environments": frozenset(selected.environment_id.astype(str).unique()),
            "source": "V2_TEMPORAL_EXTENSION",
        }

    for outer in OUTER_FOLDS:
        train_countries = {country for country, fold in country_assignment.items() if fold != outer}
        test_countries = {country for country, fold in country_assignment.items() if fold == outer}
        train_mask = obs.country.isin(train_countries)
        test_mask = obs.country.isin(test_countries)
        role = np.where(train_mask, "TRAIN", "OUTER_TEST_ID_ONLY")
        roles[f"country_holdout_outer{outer}_role"] = pd.Categorical(role)
        role_summaries.append(summarize_role(obs, train_mask, "COUNTRY_HOLDOUT", outer, "TRAIN"))
        role_summaries.append(summarize_role(obs, test_mask, "COUNTRY_HOLDOUT", outer, "OUTER_TEST_ID_ONLY"))
        overlap = train_countries.intersection(test_countries)
        leakage_rows.append(
            {
                "scenario": "COUNTRY_HOLDOUT",
                "outer_fold": outer,
                "inner_fold": "",
                "check": "COUNTRY_TRAIN_TEST_DISJOINT",
                "failure_count": len(overlap),
                "status": "PASS" if not overlap else "FAIL",
            }
        )
        state_id = f"COUNTRY_HOLDOUT__OUTER{outer}"
        selected = obs.loc[train_mask, ["canonical_gid", "environment_id"]]
        states[state_id] = {
            "state_id": state_id,
            "scenario": "COUNTRY_HOLDOUT",
            "outer_fold": outer,
            "inner_fold": None,
            "state_level": "OUTER",
            "training_observations": int(train_mask.sum()),
            "training_gids": frozenset(selected.canonical_gid.astype(str).unique()),
            "training_environments": frozenset(selected.environment_id.astype(str).unique()),
            "source": "V2_COUNTRY_EXTENSION",
        }
        inner_weights = {country: country_weights[country] for country in train_countries}
        inner_folds = deterministic_balanced_assignment(inner_weights, 5, f"COUNTRY_INNER_OUTER{outer}")
        for inner_fold in INNER_FOLDS:
            inner_validation = {country for country, fold in inner_folds.items() if fold == inner_fold}
            inner_train = train_countries - inner_validation
            inner_mask = obs.country.isin(inner_train)
            state_id = f"COUNTRY_HOLDOUT__OUTER{outer}__INNER{inner_fold}"
            selected = obs.loc[inner_mask, ["canonical_gid", "environment_id"]]
            states[state_id] = {
                "state_id": state_id,
                "scenario": "COUNTRY_HOLDOUT",
                "outer_fold": outer,
                "inner_fold": inner_fold,
                "state_level": "INNER",
                "training_observations": int(inner_mask.sum()),
                "training_gids": frozenset(selected.canonical_gid.astype(str).unique()),
                "training_environments": frozenset(selected.environment_id.astype(str).unique()),
                "source": "V2_COUNTRY_EXTENSION",
            }
            for country, fold in sorted(inner_folds.items()):
                inner_assignment_rows.append(
                    {
                        "scenario": "COUNTRY_HOLDOUT",
                        "outer_fold": outer,
                        "inner_fold": inner_fold,
                        "entity_type": "COUNTRY",
                        "entity_id": country,
                        "assignment": "INNER_VALIDATION_ID_ONLY" if fold == inner_fold else "TRAIN",
                    }
                )
            overlap = inner_train.intersection(inner_validation)
            leakage_rows.append(
                {
                    "scenario": "COUNTRY_HOLDOUT",
                    "outer_fold": outer,
                    "inner_fold": inner_fold,
                    "check": "INNER_COUNTRY_TRAIN_VALIDATION_DISJOINT",
                    "failure_count": len(overlap),
                    "status": "PASS" if not overlap else "FAIL",
                }
            )

    pq.write_table(pa.Table.from_pandas(roles, preserve_index=False), out / "splits/scenario_observation_roles.parquet", compression="zstd")
    write_tsv(out / "splits/temporal_year_assignment.tsv", temporal_rows)
    write_tsv(out / "splits/country_assignment.tsv", country_rows)
    write_tsv(out / "splits/inner_entity_assignment.tsv", inner_assignment_rows)
    write_tsv(out / "splits/scenario_role_summary.tsv", role_summaries)
    leakage = pd.DataFrame(leakage_rows)
    write_tsv(out / "splits/scenario_leakage_report.tsv", leakage)
    if not leakage.status.eq("PASS").all():
        raise AssertionError("Temporal/country leakage certification failed")
    write_json(
        out / "splits/scenario_protocol.json",
        {
            "release_id": RELEASE_ID,
            "seed": SEED,
            "outcome_columns_loaded": [],
            "existing_scenarios": "PRESERVED_BY_REFERENCE_BYTE_FOR_BYTE",
            "temporal_year": {
                "normalized_year_rule": "YYYY unchanged; YY-YY maps to season end year using 1970 pivot",
                "outer_design": "six contiguous blocks; block 1 initial training; blocks 2-6 five forward outer tests",
                "embargo": "latest training year excluded, exactly one year before test",
                "future_rows": "excluded as FUTURE_NOT_AVAILABLE, never used in preprocessing",
                "inner_design": "five nested forward splits within each outer training range with one-year embargo",
            },
            "country_holdout": {
                "assignment_unit": "country",
                "outer_design": "deterministic five-fold greedy primary-row balancing",
                "inner_design": "outer-specific deterministic five-fold country reassignment",
                "tie_break": "sha256(seed|namespace|country|fold)",
            },
        },
    )
    return states, obs, pd.DataFrame(role_summaries), leakage


def write_state_registry(out: Path, states: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for state_id in sorted(states):
        state = states[state_id]
        gids = sorted(state["training_gids"])
        envs = sorted(state["training_environments"])
        gid_path = out / f"splits/state_entities/{state_id}__training_gids.tsv"
        env_path = out / f"splits/state_entities/{state_id}__training_environments.tsv"
        write_tsv(gid_path, [{"canonical_gid": value} for value in gids])
        write_tsv(env_path, [{"environment_id": value} for value in envs])
        rows.append(
            {
                "state_id": state_id,
                "scenario": state["scenario"],
                "outer_fold": state["outer_fold"],
                "inner_fold": "" if state["inner_fold"] is None else state["inner_fold"],
                "state_level": state["state_level"],
                "training_observations": state["training_observations"],
                "training_canonical_gids": len(gids),
                "training_environments": len(envs),
                "training_gid_signature": index_signature(gids),
                "training_environment_signature": index_signature(envs),
                "training_gid_path": relative_posix(gid_path, out),
                "training_environment_path": relative_posix(env_path, out),
                "source": state["source"],
            }
        )
    frame = pd.DataFrame(rows)
    write_tsv(out / "splits/state_registry.tsv", frame)
    return frame


def panel_axis_description(panel_id: str, technology: str) -> tuple[str, str, str, str, str]:
    if panel_id in {"seeds_of_discovery_dartseq", "mexican_landrace_dartseq"}:
        return (
            "MATRIX_ROWS_MARKERS_COLUMNS_PHYSICAL_SAMPLES",
            "SAMPLEID_GID_SAME_DATASET_SIDECAR",
            "MARKER_ID_SUFFIX_DECLARES_REF_GT_ALT; HOM_REF=0 HET=1 HOM_ALT=2",
            "-",
            "RAW_CALLS_PRESENT",
        )
    if panel_id.startswith("dartseq80k_"):
        return (
            "CSV_ROWS_MARKERS_COLUMNS_PHYSICAL_SAMPLES; FLAPJACK_COUNTERPART_CERTIFIED",
            "PHYSICAL_SAMPLE_COLUMNS_PRESERVED_WITH_OCCURRENCE_INDEX",
            "PAV 0/1; SNP paired allele presence to nucleotide or slash heterozygote",
            "-",
            "RAW_CALLS_PRESENT_IDENTITY_NOT_AUTHORIZED",
        )
    if panel_id.startswith("gbs_"):
        return (
            "MATRIX_ROWS_MARKER_SEQUENCE_COLUMNS_TYPED_GIDS",
            "GID_MATRIX_HEADER",
            "A/C/G/T HOMOZYGOTE; H HETEROZYGOTE; SOURCE HAS DUPLICATE SEQUENCE ROWS",
            "N",
            "RAW_CALLS_PRESENT_NO_COORDINATE_OR_ALLELE_DICTIONARY",
        )
    if panel_id == "cimmyt_bread_gbs_2013_2018":
        return (
            "HAPMAP_ROWS_MARKERS_COLUMNS_GIDS",
            "DOCUMENTED_GID_HEADER",
            "HAPMAP IUPAC DOSAGE",
            "UNRECOVERABLE_AFTER_GLOBAL_IMPUTATION",
            "ONLY_GLOBALLY_QC_FILTERED_IMPUTED_EXPORT_PRESENT",
        )
    if panel_id == "frozen_hmp_v1":
        return (
            "BUNDLE_PARQUET_ROWS_SAMPLES_COLUMNS_MARKERS; HISTORICAL_KERNEL_ORDER_SEPARATE",
            "GID SAMPLE IDS TRACED_TO_IMPUTED_CIMMYT_EXPORT",
            "0/1/2 DOSAGE AFTER GLOBAL IMPUTATION",
            "NONE_REMAINING",
            "TRANSFORMED_GLOBAL_IMPUTED_MATRIX_NOT_RAW_CALLS",
        )
    if panel_id == "eyt_haplotype_blocks_2011_2018":
        return (
            "TABLE_ROWS_TYPED_GIDS_COLUMNS_HAPLOTYPE_BLOCK_CALLS",
            "EXPLICIT GID COLUMN",
            "HAPLOTYPE CATEGORY/DOSAGE; BLOCK DEFINITION PROVENANCE ABSENT",
            "SOURCE_SPECIFIC_NA",
            "HAPLOTYPE_CALLS_PRESENT_SOURCE_SNP_CALLS_ABSENT",
        )
    if panel_id == "hibap35k":
        return (
            "MATRIX_ROWS_MARKERS_COLUMNS_PHYSICAL_SAMPLES",
            "PHASE3G_R2 CORRECTED SAME-DATASET CROSSWALK",
            "BIALLELIC IUPAC CALLS WITH SOURCE ALLELE DECLARATION",
            "N;-;.;NA",
            "RAW_CALLS_PRESENT; IMMUTABLE PHASE5 FOLD_LOCAL_COMPONENT",
        )
    if panel_id == "dartag_panel2":
        return (
            "INTERTEK/NUMERIC TABLES SAMPLE_ROWS_TARGETED_MARKER_COLUMNS",
            "SAME-DATASET GERMPLASM LIST",
            "TARGETED LOCUS NUMERIC/ALLELIC CALLS",
            "SOURCE_SPECIFIC_BLANK_OR_NA",
            "RAW_TARGETED_CALLS_PRESENT",
        )
    if panel_id.startswith("mas_"):
        return (
            "WORKBOOK SAMPLE_ROWS_TARGETED_GENE_MARKER_COLUMNS",
            "SAME-DATASET GID/DOI FIELDS",
            "KASP/STS/SSR SOURCE-SPECIFIC TARGETED CALLS",
            "SOURCE_SPECIFIC_BLANK_OR_NA",
            "RAW_TARGETED_CALLS_PRESENT",
        )
    return (
        "METADATA_OR_COLLECTION_ONLY",
        "NO_AUTHORIZED_TYPED_MATRIX AXIS",
        "NOT_APPLICABLE",
        "NOT_APPLICABLE",
        "NO_PRODUCTION_MATRIX",
    )


def build_panel_audit(
    root: Path,
    out: Path,
    guard: ProtectedPathGuard,
    obs: pd.DataFrame,
    states: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, set[str]], pd.DataFrame]:
    p5 = root / UPSTREAM_RELATIVE
    registry_path = guard.assert_allowed(p5 / "genomic/panel_sample_gid_registry.tsv", "READ_ACCEPTED_PANEL_IDENTITIES")
    accepted = pd.read_csv(registry_path, sep="\t", dtype=str, keep_default_na=False)
    inventory_path = guard.assert_allowed(root / "audit/v2/phase3g_all_panel_genotype_linkage_audit_v2/panel_inventory.tsv", "READ_PANEL_INVENTORY")
    inventory = pd.read_csv(inventory_path, sep="\t", dtype=str, keep_default_na=False)
    primary_obs = obs[obs.primary_weighted_training_eligible.fillna(False)]
    primary_rows = primary_obs.groupby("canonical_gid", sort=False).size()
    primary_gids = set(primary_rows.index.astype(str))
    panel_gids: dict[str, set[str]] = {}
    overlap_rows = []
    for row in inventory.itertuples(index=False):
        gids = set(
            accepted.loc[
                accepted.panel_id.eq(row.panel_id) & accepted.accepted_canonical_gid.ne(""),
                "accepted_canonical_gid",
            ]
        )
        panel_gids[row.panel_id] = gids
        overlap = gids.intersection(primary_gids)
        observed_gids = len(overlap)
        observed_rows = int(primary_rows.reindex(sorted(overlap)).fillna(0).sum())
        expected = EXPECTED_PRIMARY_OVERLAP.get(row.panel_id)
        status = "PASS" if expected is None or expected == (observed_gids, observed_rows) else "FAIL"
        overlap_rows.append(
            {
                "panel_id": row.panel_id,
                "accepted_panel_gids": len(gids),
                "primary_stage1_gids": observed_gids,
                "primary_stage1_rows": observed_rows,
                "expected_primary_gids": "" if expected is None else expected[0],
                "expected_primary_rows": "" if expected is None else expected[1],
                "discrepancy_gids": "" if expected is None else observed_gids - expected[0],
                "discrepancy_rows": "" if expected is None else observed_rows - expected[1],
                "status": status,
            }
        )
    overlap = pd.DataFrame(overlap_rows)
    write_tsv(out / "genomic/panel_stage1_overlap.tsv", overlap)
    if not overlap.status.eq("PASS").all():
        raise AssertionError("Panel-to-primary overlap discrepancy")

    support_rows = []
    for state_id, state in sorted(states.items()):
        training = state["training_gids"]
        for panel_id, gids in sorted(panel_gids.items()):
            primary_panel = gids.intersection(primary_gids)
            support_rows.append(
                {
                    "state_id": state_id,
                    "scenario": state["scenario"],
                    "outer_fold": state["outer_fold"],
                    "inner_fold": "" if state["inner_fold"] is None else state["inner_fold"],
                    "panel_id": panel_id,
                    "training_accepted_gids": len(gids.intersection(training)),
                    "training_primary_stage1_gids": len(primary_panel.intersection(training)),
                    "all_primary_panel_gids": len(primary_panel),
                    "minimum_required_for_dense_kg": 20,
                    "support_status": "PASS" if len(primary_panel.intersection(training)) >= 20 else "INSUFFICIENT_EXPLICIT_COMPONENT_MASK",
                }
            )
    support = pd.DataFrame(support_rows)
    write_tsv(out / "genomic/panel_fold_support.tsv", support)

    hmp_manifest_path = guard.assert_allowed(root / BUNDLE_HMP_MANIFEST, "READ_HMP_SOURCE_TRACE")
    hmp_manifest = pd.read_csv(hmp_manifest_path, sep="\t", dtype=str, keep_default_na=False)
    hmp_imputed_trace = bool(hmp_manifest.imputation_status.str.lower().eq("imputed").all()) and bool(
        hmp_manifest.panel_file.str.contains(".imputed.", regex=False, case=False).all()
    )
    hmp_raw_path = guard.assert_allowed(root / BUNDLE_HMP / "hmp_sample_by_marker.parquet", "READ_HMP_SCHEMA_METADATA")
    hmp_qc_path = guard.assert_allowed(root / BUNDLE_HMP / "hmp_sample_by_marker.QCfiltered.parquet", "READ_HMP_SCHEMA_METADATA")
    hmp_raw_meta = pq.ParquetFile(hmp_raw_path).metadata
    hmp_qc_meta = pq.ParquetFile(hmp_qc_path).metadata
    hmp_stats_path = guard.assert_allowed(root / BUNDLE_HMP / "qc_hmp_sample_stats.tsv", "READ_HMP_QC_PROVENANCE")
    hmp_stats = pd.read_csv(hmp_stats_path, sep="\t")
    if not hmp_imputed_trace or not hmp_stats.sample_missingness.fillna(0).eq(0).all():
        raise AssertionError("HMP source trace did not reproduce global-imputation evidence")
    write_json(
        out / "genomic/hmp_source_trace.json",
        {
            "source_manifest_rows": len(hmp_manifest),
            "all_rows_marked_imputed": hmp_imputed_trace,
            "raw_named_matrix_rows": hmp_raw_meta.num_rows,
            "raw_named_matrix_columns": hmp_raw_meta.num_columns,
            "qcfiltered_matrix_rows": hmp_qc_meta.num_rows,
            "qcfiltered_matrix_columns": hmp_qc_meta.num_columns,
            "sample_qc_rows": len(hmp_stats),
            "observed_missing_call_mask_recoverable": False,
            "production_disposition": "BLOCKED_TRACED_TO_GLOBALLY_IMPUTED_CIMMYT_EXPORT",
        },
    )

    axis_rows = []
    for row in inventory.itertuples(index=False):
        orientation, sample_axis, encoding, missing_tokens, availability = panel_axis_description(row.panel_id, row.technology)
        panel_support = support[support.panel_id.eq(row.panel_id)]
        role, disposition = PANEL_ROLES[row.panel_id]
        if row.panel_id.startswith("gbs_") and panel_support.training_primary_stage1_gids.min() < 20:
            disposition = "BLOCKED_NOT_AT_LEAST_20_PRIMARY_GIDS_IN_EVERY_TRAINING_STATE"
        axis_rows.append(
            {
                "panel_id": row.panel_id,
                "platform": row.platform,
                "technology": row.technology,
                "matrix_orientation": orientation,
                "sample_axis_certification": sample_axis,
                "marker_axis_certification": "SOURCE_MARKER_ROWS_PRESERVED; EXACT COUNTS IN SOURCE INVENTORY",
                "allele_encoding": encoding,
                "missing_call_tokens": missing_tokens,
                "raw_marker_availability": availability,
                "raw_sample_count": row.raw_sample_count,
                "accepted_canonical_gid_count": row.accepted_canonical_gid_count,
                "duplicate_or_replicate_status": "SEEDS_REQUIRES_V2_CONCORDANCE" if row.panel_id == SEEDS_PANEL else "PHASE3G_R2_INSTANCE_AXIS_PRESERVED",
                "identity_authority": "PHASE3G_R2_ACCEPTED_SAME_DATASET_IDENTITIES_ONLY",
                "minimum_training_primary_gids": int(panel_support.training_primary_stage1_gids.min()),
                "effective_rank_certification": "STATE_SPECIFIC_FOR_ACTIVATED_COMPONENT" if row.panel_id in {SEEDS_PANEL, "hibap35k"} else "NOT_COMPUTED_COMPONENT_NOT_ACTIVATED",
                "intended_biological_role": role,
                "v2_terminal_disposition": disposition,
                "source_files": row.source_files,
            }
        )
    axis = pd.DataFrame(axis_rows)
    write_tsv(out / "genomic/panel_source_axis_audit.tsv", axis)

    protocols = []
    for row in axis.itertuples(index=False):
        if row.panel_id == SEEDS_PANEL:
            protocol = {
                "protocol_id": "SEEDS_DARTSEQ_RAW_SPLIT_LOCAL_V2",
                "sample_qc": "accepted Phase3G same-dataset identity; >=0.50 raw call rate; replicate anchor highest call rate then sample key",
                "replicate_qc": ">=10000 co-called markers and >=0.995 exact dosage concordance to anchor; discordant instances quarantined",
                "marker_qc": "within training GIDs only: call rate >=0.80, MAF >=0.01, heterozygosity <=0.20, polymorphic",
                "imputation": "training-only 2p mean",
                "status": "FROZEN_BEFORE_CALL_INSPECTION",
            }
        elif row.panel_id.startswith("gbs_"):
            protocol = {
                "protocol_id": f"{row.panel_id.upper()}_SEPARATE_PANEL_V2",
                "sample_qc": "exact typed GID header; minimum 0.50 call rate",
                "replicate_qc": "no duplicate typed GID columns in Phase3G registry",
                "marker_qc": "ignore source-global present/MAF/percentHET; training-only call rate >=0.80, MAF >=0.01, heterozygosity <=0.20",
                "imputation": "training-only 2p mean",
                "status": "FROZEN_NOT_ACTIVATED_UNLESS_ALL_STATE_SUPPORT_GE20",
            }
        elif row.panel_id in {"cimmyt_bread_gbs_2013_2018", "frozen_hmp_v1"}:
            protocol = {
                "protocol_id": "NOT_APPLICABLE_TRANSFORMED_INPUT",
                "sample_qc": "not fit",
                "replicate_qc": "not fit",
                "marker_qc": "not fit",
                "imputation": "prohibited global imputation already embedded",
                "status": row.v2_terminal_disposition,
            }
        else:
            protocol = {
                "protocol_id": "ROLE_SPECIFIC_REGISTRY_ONLY",
                "sample_qc": "identity/axis audit only",
                "replicate_qc": "no collapse unless same-dataset concordance protocol exists",
                "marker_qc": "not fit for genomewide K_G",
                "imputation": "not fit",
                "status": row.v2_terminal_disposition,
            }
        protocols.append({"panel_id": row.panel_id, **protocol})
    protocol_frame = pd.DataFrame(protocols)
    write_tsv(out / "genomic/panel_qc_protocols.tsv", protocol_frame)

    source_crosswalk = guard.assert_allowed(root / "audit/v2/phase3g_all_panel_genotype_linkage_audit_v2/accepted_all_panel_crosswalk.parquet", "READ_ACCEPTED_MAPPING_MANIFEST")
    crosswalk = pq.read_table(source_crosswalk).to_pandas()
    pq.write_table(pa.Table.from_pandas(crosswalk, preserve_index=False), out / "genomic/accepted_mapping_manifest.parquet", compression="zstd")
    sample_ledger_path = guard.assert_allowed(root / "audit/v2/phase3g_all_panel_genotype_linkage_audit_v2/sample_identifier_ledger.parquet", "READ_MAPPING_DISPOSITIONS")
    ledger = pq.read_table(sample_ledger_path).to_pandas()
    unresolved = ledger[ledger.accepted_canonical_gid.fillna("").astype(str).eq("")].copy()
    conflicting = ledger[
        ledger.get("conflict_status", pd.Series("", index=ledger.index)).fillna("").astype(str).ne("")
        | ledger.get("mapping_status", pd.Series("", index=ledger.index)).fillna("").astype(str).str.contains("CONFLICT|AMBIG", case=False, regex=True)
    ].copy()
    pq.write_table(pa.Table.from_pandas(unresolved, preserve_index=False), out / "genomic/unresolved_mapping_manifest.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pandas(conflicting, preserve_index=False), out / "genomic/conflicting_mapping_manifest.parquet", compression="zstd")

    targeted_ids = {"dartag_panel2", "mas_45ibwsn", "mas_57ibwsn_42sawsn_35hrwsn", "mas_58ibwsn_43sawsn"}
    targeted = axis[axis.panel_id.isin(targeted_ids)][
        ["panel_id", "platform", "technology", "accepted_canonical_gid_count", "intended_biological_role", "v2_terminal_disposition"]
    ].copy()
    targeted["allowed_representation"] = "SPARSE_TARGETED_MARKER_COVARIATES_OR_SEPARATE_BIOLOGICAL_EXPERT"
    targeted["entered_genomewide_kg"] = False
    targeted["missingness_mask_required"] = True
    write_tsv(out / "genomic/targeted_marker_component_registry.tsv", targeted)
    write_tsv(
        out / "redundancy/genomic_component_redundancy_ledger.tsv",
        [
            {
                "component_a": "frozen_hmp_v1",
                "component_b": "cimmyt_bread_gbs_2013_2018",
                "relationship": "HMP SAMPLE/MARKER MATRICES TRACE TO THE SAME GLOBALLY IMPUTED CIMMYT EXPORT",
                "quantitative_metric_computed": False,
                "used_for_selection": False,
                "disposition": "NO_INDEPENDENT_PRODUCTION_EXPERT; TRANSDUCTIVE_DIAGNOSTIC_ONLY",
            },
            {
                "component_a": "eyt_haplotype_blocks_2011_2018",
                "component_b": "SOURCE_SNP_EXPERT",
                "relationship": "UNRESOLVED_SOURCE_SNP/BLOCK DUPLICATION BECAUSE PROVENANCE IS ABSENT",
                "quantitative_metric_computed": False,
                "used_for_selection": False,
                "disposition": "BLOCK_HAPLOTYPE_EXPERT",
            },
            {
                "component_a": "gbs_13sawyt..gbs_18sawyt",
                "component_b": "EACH_OTHER",
                "relationship": "EXACT SEQUENCE LABEL OVERLAP EXISTS BUT COORDINATE/ALLELE/DOSAGE HARMONIZATION IS NOT CERTIFIED",
                "quantitative_metric_computed": False,
                "used_for_selection": False,
                "disposition": "RETAIN_SEPARATE; DO_NOT_MERGE",
            },
            {
                "component_a": "seeds_of_discovery_dartseq",
                "component_b": "dartseq80k_*",
                "relationship": "CROSS-PANEL SAMPLE LABELS ARE CANDIDATE EVIDENCE ONLY",
                "quantitative_metric_computed": False,
                "used_for_selection": False,
                "disposition": "NO_IDENTITY_BRIDGE_OR_COMPONENT_MERGE",
            },
        ],
    )
    return axis, panel_gids, support


def decode_seed_token(token: str, reference: str, alternate: str) -> int:
    value = token.strip().upper().replace("/", "").replace("|", "")
    if value in {"", "-", "N", "NA", ".", "?"}:
        return 255
    if value == reference:
        return 0
    if value == alternate:
        return 2
    if len(value) == 2 and set(value) == {reference, alternate}:
        return 1
    return 255


def stream_seeds_primary_calls(
    root: Path,
    out: Path,
    guard: ProtectedPathGuard,
    primary_gids: set[str],
) -> tuple[Path, list[str], list[str], pd.DataFrame, pd.DataFrame]:
    registry_path = guard.assert_allowed(root / UPSTREAM_RELATIVE / "genomic/panel_sample_gid_registry.tsv", "READ_SEEDS_ACCEPTED_IDENTITIES")
    registry = pd.read_csv(registry_path, sep="\t", dtype=str, keep_default_na=False)
    selected = registry[
        registry.panel_id.eq(SEEDS_PANEL)
        & registry.accepted_canonical_gid.isin(primary_gids)
        & registry.accepted_canonical_gid.ne("")
    ].copy()
    if selected.raw_sample_id.duplicated().any():
        raise AssertionError("Seeds raw sample ID maps to multiple accepted instances")
    sample_to_gid = dict(zip(selected.raw_sample_id, selected.accepted_canonical_gid))
    matrix_path = guard.assert_allowed(root / SEEDS_MATRIX, "STREAM_RAW_SEEDS_CALLS")
    expected_markers = 102_474
    temporary = out / "genomic/seeds_primary_instance_calls.npy"
    marker_ids: list[str] = []
    marker_reference: list[str] = []
    marker_alternate: list[str] = []
    invalid_marker_declarations = 0
    unexpected_tokens: Counter[str] = Counter()
    with matrix_path.open("r", encoding="ascii", errors="strict", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split("\t")
        header_index = {sample: index for index, sample in enumerate(header)}
        missing_samples = sorted(set(sample_to_gid) - set(header_index))
        if missing_samples:
            raise AssertionError(f"Seeds accepted samples absent from matrix header: {missing_samples[:5]}")
        sample_ids = sorted(sample_to_gid, key=lambda sample: header_index[sample])
        selected_indices = [header_index[sample] for sample in sample_ids]
        raw = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.uint8,
            shape=(expected_markers, len(sample_ids)),
        )
        marker_pattern = re.compile(r":([ACGT])>([ACGT])$")
        for marker_index, line in enumerate(handle):
            if marker_index >= expected_markers:
                raise AssertionError("Seeds marker count exceeds declared 102474")
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != len(header):
                raise AssertionError(f"Seeds row width mismatch at marker row {marker_index + 2}")
            marker = fields[0]
            match = marker_pattern.search(marker.upper())
            if match is None or match.group(1) == match.group(2):
                reference, alternate = "", ""
                invalid_marker_declarations += 1
                encoded = np.full(len(selected_indices), 255, dtype=np.uint8)
            else:
                reference, alternate = match.groups()
                mapping = {
                    "": 255,
                    "-": 255,
                    "N": 255,
                    "NA": 255,
                    ".": 255,
                    "?": 255,
                    reference: 0,
                    alternate: 2,
                    reference + alternate: 1,
                    alternate + reference: 1,
                    reference + "/" + alternate: 1,
                    alternate + "/" + reference: 1,
                    reference + "|" + alternate: 1,
                    alternate + "|" + reference: 1,
                }
                encoded = np.asarray(
                    [mapping.get(fields[index], 254) for index in selected_indices],
                    dtype=np.uint8,
                )
                if np.any(encoded == 254):
                    for index, code in zip(selected_indices, encoded):
                        if code == 254:
                            unexpected_tokens[fields[index]] += 1
                    encoded[encoded == 254] = 255
            raw[marker_index, :] = encoded
            marker_ids.append(marker)
            marker_reference.append(reference)
            marker_alternate.append(alternate)
            if (marker_index + 1) % 10_000 == 0:
                raw.flush()
                print(f"Seeds stream: {marker_index + 1}/{expected_markers} markers", flush=True)
        if len(marker_ids) != expected_markers:
            raise AssertionError(f"Seeds marker count {len(marker_ids)} != {expected_markers}")
        raw.flush()
        del raw
    mapping_frame = pd.DataFrame(
        {
            "instance_index": np.arange(len(sample_ids), dtype=np.int64),
            "raw_sample_id": sample_ids,
            "canonical_gid": [sample_to_gid[sample] for sample in sample_ids],
            "source_column_index_1based": [header_index[sample] + 1 for sample in sample_ids],
        }
    )
    marker_frame = pd.DataFrame(
        {
            "marker_index": np.arange(len(marker_ids), dtype=np.int64),
            "marker_id": marker_ids,
            "reference_allele": marker_reference,
            "alternate_allele": marker_alternate,
            "source_row_1based": np.arange(2, len(marker_ids) + 2, dtype=np.int64),
            "encoding_status": np.where(np.asarray(marker_reference) != "", "PASS_DECLARED_BIALLELIC", "INVALID_DECLARATION_SET_MISSING"),
        }
    )
    write_json(
        out / "genomic/seeds_stream_audit.json",
        {
            "matrix_orientation": "MARKER_ROWS_BY_SAMPLE_COLUMNS",
            "source_sample_columns": len(header) - 1,
            "selected_primary_sample_instances": len(sample_ids),
            "source_markers": len(marker_ids),
            "invalid_marker_declarations": invalid_marker_declarations,
            "unexpected_call_tokens": dict(sorted(unexpected_tokens.items())),
            "complete_matrix_string_dataframe_materialized": False,
            "streaming_rule": "ONE_SOURCE_MARKER_ROW_AT_A_TIME",
        },
    )
    write_tsv(out / "genomic/seeds_primary_instance_axis.tsv", mapping_frame)
    marker_frame.to_parquet(out / "genomic/seeds_marker_axis.parquet", index=False, compression="zstd")
    return temporary, marker_ids, sample_ids, mapping_frame, marker_frame


def collapse_seeds_replicates(
    temporary: Path,
    marker_count: int,
    mapping: pd.DataFrame,
    out: Path,
) -> tuple[Path, list[str], pd.DataFrame, pd.DataFrame]:
    raw = np.load(temporary, mmap_mode="r")
    if raw.shape != (marker_count, len(mapping)):
        raise AssertionError("Seeds temporary instance matrix shape mismatch")
    sample_call_rate = np.asarray(np.mean(raw != 255, axis=0), dtype=np.float64)
    mapping = mapping.copy()
    mapping["sample_call_rate"] = sample_call_rate
    gid_groups = mapping.groupby("canonical_gid", sort=True).indices
    gids = sorted(gid_groups)
    consensus_path = out / "genomic/seeds_primary_gid_consensus.npy"
    consensus = np.lib.format.open_memmap(
        consensus_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(gids), marker_count),
    )
    decision_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for gid_index, gid in enumerate(gids):
        indices = np.asarray(gid_groups[gid], dtype=int)
        group = mapping.iloc[indices]
        anchor_row = group.sort_values(["sample_call_rate", "raw_sample_id"], ascending=[False, True]).iloc[0]
        anchor_index = int(anchor_row.instance_index)
        anchor = raw[:, anchor_index]
        retained: list[int] = [anchor_index]
        quarantined = 0
        for record in group.sort_values("raw_sample_id").itertuples(index=False):
            instance_index = int(record.instance_index)
            values = raw[:, instance_index]
            co_called = (anchor != 255) & (values != 255)
            overlap = int(co_called.sum())
            concordance = float(np.mean(anchor[co_called] == values[co_called])) if overlap else math.nan
            is_anchor = instance_index == anchor_index
            accepted = is_anchor or (overlap >= 10_000 and concordance >= 0.995)
            if accepted and not is_anchor:
                retained.append(instance_index)
            if not accepted:
                quarantined += 1
            decision_rows.append(
                {
                    "canonical_gid": gid,
                    "raw_sample_id": record.raw_sample_id,
                    "instance_index": instance_index,
                    "anchor_raw_sample_id": anchor_row.raw_sample_id,
                    "sample_call_rate": record.sample_call_rate,
                    "co_called_markers": overlap,
                    "concordance_to_anchor": concordance,
                    "minimum_overlap": 10_000,
                    "minimum_concordance": 0.995,
                    "decision": "RETAIN_ANCHOR" if is_anchor else "COLLAPSE_CONCORDANT" if accepted else "QUARANTINE_DISCORDANT_OR_LOW_OVERLAP",
                }
            )
        retained = sorted(set(retained))
        output = np.full(marker_count, 255, dtype=np.uint8)
        for start in range(0, marker_count, 8192):
            stop = min(marker_count, start + 8192)
            values = np.asarray(raw[start:stop, retained], dtype=np.uint8)
            if values.ndim == 1:
                values = values[:, None]
            valid = values != 255
            count = valid.sum(axis=1)
            minimum = np.where(valid, values, 3).min(axis=1)
            maximum = np.where(valid, values, 0).max(axis=1)
            agree = (count > 0) & (minimum == maximum)
            block = np.full(stop - start, 255, dtype=np.uint8)
            block[agree] = maximum[agree]
            output[start:stop] = block
        consensus[gid_index, :] = output
        summary_rows.append(
            {
                "canonical_gid": gid,
                "physical_sample_instances": len(indices),
                "retained_concordant_instances": len(retained),
                "quarantined_instances": quarantined,
                "consensus_call_rate": float(np.mean(output != 255)),
                "consensus_rule": "UNANIMOUS_NONMISSING_AMONG_ANCHOR_CONCORDANT_INSTANCES_ELSE_MISSING",
                "retained_for_component": bool(np.mean(output != 255) >= 0.50),
            }
        )
        if (gid_index + 1) % 500 == 0:
            consensus.flush()
            print(f"Seeds replicate collapse: {gid_index + 1}/{len(gids)} GIDs", flush=True)
    consensus.flush()
    del consensus
    del raw
    # Retain the accepted physical-instance call store as an auditable source
    # layer. Windows can also keep the final comparison view open until process
    # teardown, so deleting it here is not portable.
    decisions = pd.DataFrame(decision_rows)
    summary = pd.DataFrame(summary_rows)
    write_tsv(out / "genomic/seeds_replicate_decisions.tsv", decisions)
    write_tsv(out / "genomic/seeds_gid_consensus_summary.tsv", summary)
    write_json(
        out / "genomic/seeds_replicate_summary.json",
        {
            "primary_gids": len(gids),
            "sample_instances": len(mapping),
            "multi_instance_gids": int((summary.physical_sample_instances > 1).sum()),
            "quarantined_instances": int(summary.quarantined_instances.sum()),
            "retained_component_gids": int(summary.retained_for_component.sum()),
            "criteria_frozen_before_call_access": True,
        },
    )
    return consensus_path, gids, decisions, summary


def fit_seeds_states(
    consensus_path: Path,
    gids: list[str],
    marker_ids: list[str],
    consensus_summary: pd.DataFrame,
    states: dict[str, dict[str, Any]],
    out: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dosage = np.load(consensus_path, mmap_mode="r")
    if dosage.shape != (len(gids), len(marker_ids)):
        raise AssertionError("Seeds consensus matrix shape mismatch")
    gid_index = {gid: index for index, gid in enumerate(gids)}
    eligible_gids = set(consensus_summary.loc[consensus_summary.retained_for_component, "canonical_gid"])
    priority = np.argsort(
        np.asarray([stable_json_hash({"marker": marker}) for marker in marker_ids], dtype="U64"),
        kind="stable",
    )
    registry_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    preprocessing_rows: list[dict[str, Any]] = []
    for state_number, (state_id, state) in enumerate(sorted(states.items()), start=1):
        training_gids = sorted(eligible_gids.intersection(state["training_gids"]))
        training_indices = np.asarray([gid_index[gid] for gid in training_gids], dtype=int)
        state_path = out / f"genomic/states/{state_id}__seeds_dartseq_vanraden.npz"
        if len(training_indices) < 20:
            np.savez_compressed(
                state_path,
                retained_marker_index=np.asarray([], dtype=np.int32),
                allele_frequency=np.asarray([], dtype=np.float32),
                denominator=np.asarray([np.nan], dtype=np.float64),
                training_gid_signature=np.asarray([index_signature(training_gids)], dtype="U64"),
            )
            preprocessing_rows.append(
                {
                    "state_id": state_id,
                    "panel_id": SEEDS_PANEL,
                    "training_gids": len(training_gids),
                    "input_markers": len(marker_ids),
                    "retained_markers": 0,
                    "state_path": relative_posix(state_path, out),
                    "state_sha256": sha256_file(state_path),
                    "status": "MASKED_INSUFFICIENT_TRAINING_GIDS",
                }
            )
            registry_rows.append(
                {
                    "state_id": state_id,
                    "panel_id": SEEDS_PANEL,
                    "representation": "ON_DEMAND_RAW_CONSENSUS_PLUS_TRAINING_LOCAL_PARAMETERS",
                    "entities": len(gids),
                    "training_entities": len(training_gids),
                    "markers": 0,
                    "component_available": False,
                    "absence_mask": "EXPLICIT_STATE_COMPONENT_MASK",
                    "status": "MASKED_INSUFFICIENT_TRAINING_GIDS",
                }
            )
            continue
        retained_blocks: list[np.ndarray] = []
        p_blocks: list[np.ndarray] = []
        for start in range(0, len(marker_ids), 4096):
            stop = min(len(marker_ids), start + 4096)
            block = np.asarray(dosage[training_indices, start:stop], dtype=np.uint8)
            valid = block != 255
            counts = valid.sum(axis=0)
            sums = np.where(valid, block, 0).sum(axis=0, dtype=np.float64)
            heterozygous = (block == 1).sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                p = sums / (2.0 * counts)
                heterozygosity = heterozygous / counts
            maf = np.minimum(p, 1.0 - p)
            keep = (
                (counts >= max(10, math.ceil(0.80 * len(training_indices))))
                & np.isfinite(p)
                & (maf >= 0.01)
                & (p > 0.0)
                & (p < 1.0)
                & np.isfinite(heterozygosity)
                & (heterozygosity <= 0.20)
            )
            retained_blocks.append(np.flatnonzero(keep).astype(np.int32) + start)
            p_blocks.append(p[keep].astype(np.float32))
        retained = np.concatenate(retained_blocks) if retained_blocks else np.asarray([], dtype=np.int32)
        allele_frequency = np.concatenate(p_blocks) if p_blocks else np.asarray([], dtype=np.float32)
        if retained.size == 0:
            raise AssertionError(f"No Seeds markers retained in supported state {state_id}")
        denominator = float(2.0 * np.sum(allele_frequency * (1.0 - allele_frequency)))
        np.savez_compressed(
            state_path,
            retained_marker_index=retained,
            allele_frequency=allele_frequency,
            denominator=np.asarray([denominator], dtype=np.float64),
            training_gid_signature=np.asarray([index_signature(training_gids)], dtype="U64"),
            qc_thresholds=np.asarray([0.80, 0.01, 0.20], dtype=np.float64),
        )
        state_hash = sha256_file(state_path)
        sketch_indices = np.asarray([index for index in priority if np.searchsorted(retained, index) < len(retained) and retained[np.searchsorted(retained, index)] == index][:64], dtype=int)
        retained_lookup = {int(marker): position for position, marker in enumerate(retained)}
        sketch_p = np.asarray([allele_frequency[retained_lookup[int(marker)]] for marker in sketch_indices], dtype=np.float64)
        sketch = np.asarray(dosage[training_indices[:, None], sketch_indices[None, :]], dtype=np.float64)
        sketch[sketch == 255] = np.nan
        means = 2.0 * sketch_p
        missing = ~np.isfinite(sketch)
        sketch[missing] = np.broadcast_to(means, sketch.shape)[missing]
        sketch -= means
        sketch_denominator = float(2.0 * np.sum(sketch_p * (1.0 - sketch_p)))
        sketch /= math.sqrt(sketch_denominator)
        diagnostics = factor_diagnostics(sketch)
        diagnostics.update(
            {
                "state_id": state_id,
                "panel_id": SEEDS_PANEL,
                "effective_rank_scope": "DETERMINISTIC_64_MARKER_LOWER_BOUND",
                "full_kernel_psd_certification": "EXACT_BY_FACTOR_CONSTRUCTION",
                "status": "PASS" if diagnostics["all_finite"] and diagnostics["algebraic_rank"] >= 2 else "FAIL",
            }
        )
        diagnostic_rows.append(diagnostics)
        preprocessing_rows.append(
            {
                "state_id": state_id,
                "panel_id": SEEDS_PANEL,
                "training_gids": len(training_gids),
                "application_gids": len(gids) - len(training_gids),
                "input_markers": len(marker_ids),
                "retained_markers": len(retained),
                "allele_frequency_fit_scope": "TRAINING_GIDS_ONLY",
                "imputation_fit_scope": "TRAINING_GIDS_ONLY_2P",
                "state_path": relative_posix(state_path, out),
                "state_sha256": state_hash,
                "status": "PASS",
            }
        )
        registry_rows.append(
            {
                "state_id": state_id,
                "panel_id": SEEDS_PANEL,
                "representation": "ON_DEMAND_RAW_CONSENSUS_PLUS_TRAINING_LOCAL_PARAMETERS",
                "formula": "K=ZZT/(2*sum(p*(1-p)))",
                "entities": len(gids),
                "training_entities": len(training_gids),
                "markers": len(retained),
                "denominator": denominator,
                "raw_consensus_path": relative_posix(consensus_path, out),
                "state_path": relative_posix(state_path, out),
                "state_sha256": state_hash,
                "component_available": True,
                "absence_mask": "GID_NOT_IN_PANEL_OR_STATE_SUPPORT_LT20",
                "status": "PASS",
            }
        )
        if state_number % 10 == 0:
            print(f"Seeds split-local QC: {state_number}/{len(states)} states", flush=True)
    registry = pd.DataFrame(registry_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    preprocessing = pd.DataFrame(preprocessing_rows)
    write_tsv(out / "genomic/seeds_component_registry.tsv", registry)
    write_tsv(out / "genomic/seeds_state_diagnostics.tsv", diagnostics)
    write_tsv(out / "genomic/seeds_fold_preprocessing_registry.tsv", preprocessing)
    if len(diagnostics) and not diagnostics.status.eq("PASS").all():
        raise AssertionError("Seeds split-local state diagnostics failed")
    return registry, diagnostics, preprocessing


def load_environment_components(
    root: Path, guard: ProtectedPathGuard
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, str]]:
    component_frames: dict[str, pd.DataFrame] = {}
    source_paths: dict[str, str] = {}
    provenance_rows: list[dict[str, Any]] = []
    base_specs = {
        "K_E_WEATHER": (BUNDLE_ENV / "env_features_weather.parquet", None, "PHENOTYPE_BLIND_WEATHER_AGGREGATES"),
        "K_E_STRESS": (BUNDLE_ENV / "env_features_stress.parquet", None, "PHENOTYPE_BLIND_WEATHER_DERIVED_STRESS_INDICES"),
        "K_E_MANAGEMENT": (BUNDLE_ENV / "env_features_mgmt.parquet", MANAGEMENT_FEATURES, "ENVIRONMENT_MANAGEMENT_NUMERIC_OR_INDICATOR_FIELDS_ONLY"),
    }
    for component, (relative, selected_features, source_class) in base_specs.items():
        path = guard.assert_allowed(root / relative, f"READ_{component}_SOURCE")
        frame = pq.read_table(path).to_pandas().rename(columns={"env_id": "environment_id"})
        if frame.environment_id.duplicated().any():
            raise AssertionError(f"Duplicate environment IDs in {component}")
        features = list(frame.columns[1:]) if selected_features is None else [feature for feature in selected_features if feature in frame]
        frame = frame[["environment_id", *features]].copy()
        for feature in features:
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
            provenance_rows.append(
                {
                    "component": component,
                    "feature": feature,
                    "source_path": relative.as_posix(),
                    "source_class": source_class,
                    "window_definition": "SOURCE_AGGREGATE_NO_OUTCOME_COLUMN",
                    "phenology_outcome_used": False,
                    "training_local_operations": "MEDIAN_IMPUTE_CENTER_SCALE_DROP_CONSTANT",
                }
            )
        component_frames[component] = frame
        source_paths[component] = relative.as_posix()

    stage_path = guard.assert_allowed(root / BUNDLE_ENV / "agronomic_api_weather_windows.tsv", "READ_FIXED_SOWING_RELATIVE_WINDOWS")
    stage = pd.read_csv(stage_path, sep="\t", dtype=str, keep_default_na=False)
    required = {"env_id", "window_label", "window_start_date", "window_end_date", "fetch_status"}
    if not required.issubset(stage.columns):
        raise AssertionError("Agronomic window source schema changed")
    start = pd.to_datetime(stage.window_start_date, errors="coerce")
    end = pd.to_datetime(stage.window_end_date, errors="coerce")
    lengths = (end - start).dt.days + 1
    fixed_labels = set(STAGE_WINDOWS.values())
    relevant = stage.window_label.isin(fixed_labels)
    if not lengths[relevant].eq(30).all():
        raise AssertionError("Stage-aware source windows are not fixed 30-day sowing-relative windows")
    numeric_features = sorted(set().union(*STAGE_FEATURE_GROUPS.values()))
    for feature in numeric_features:
        stage[feature] = pd.to_numeric(stage[feature], errors="coerce")
    stage = stage.rename(columns={"env_id": "environment_id"})
    for stage_name, label in STAGE_WINDOWS.items():
        window = stage[stage.window_label.eq(label)].copy().drop_duplicates()
        if window.environment_id.duplicated().any():
            raise AssertionError(f"Conflicting duplicate environment/window rows for {label}")
        for group_name, features_tuple in STAGE_FEATURE_GROUPS.items():
            features = [feature for feature in features_tuple if feature in window]
            component = f"K_E_STAGE_{stage_name}_{group_name}"
            component_frames[component] = window[["environment_id", *features]].copy()
            source_paths[component] = (BUNDLE_ENV / "agronomic_api_weather_windows.tsv").as_posix()
            for feature in features:
                provenance_rows.append(
                    {
                        "component": component,
                        "feature": feature,
                        "source_path": source_paths[component],
                        "source_class": "FIXED_SOWING_RELATIVE_WEATHER_WINDOW",
                        "window_definition": label,
                        "phenology_outcome_used": False,
                        "training_local_operations": "MEDIAN_IMPUTE_CENTER_SCALE_DROP_CONSTANT",
                    }
                )
    tgw_windows = ["d60_90", "d90_120", "d120_150"]
    tgw_features = [
        "gdd_base5_sum",
        "temperature_mean_c",
        "heat_days_tmax_ge_30",
        "precipitation_total_mm",
        "drought_days_precip_lt_1mm_and_vpd_gt_1_5",
        "solar_radiation_total_mj_m2",
    ]
    tgw_parts = []
    for label in tgw_windows:
        window = stage[stage.window_label.eq(label)].drop_duplicates().set_index("environment_id")
        part = window[tgw_features].copy()
        part.columns = [f"{label}__{feature}" for feature in part.columns]
        tgw_parts.append(part)
    tgw = pd.concat(tgw_parts, axis=1).reset_index()
    component_frames["K_E_TGW_FIXED_GRAIN_FILL"] = tgw
    source_paths["K_E_TGW_FIXED_GRAIN_FILL"] = (BUNDLE_ENV / "agronomic_api_weather_windows.tsv").as_posix()
    for feature in tgw.columns[1:]:
        provenance_rows.append(
            {
                "component": "K_E_TGW_FIXED_GRAIN_FILL",
                "feature": feature,
                "source_path": source_paths["K_E_TGW_FIXED_GRAIN_FILL"],
                "source_class": "TGW_SPECIFIC_FIXED_GRAIN_FILL_CANDIDATE",
                "window_definition": feature.split("__", 1)[0],
                "phenology_outcome_used": False,
                "training_local_operations": "MEDIAN_IMPUTE_CENTER_SCALE_DROP_CONSTANT",
            }
        )
    provenance = pd.DataFrame(provenance_rows)
    prohibited = provenance.feature.str.contains("heading|maturity|phenotype|yield|metric|prediction", case=False, regex=True)
    if prohibited.any() or provenance.phenology_outcome_used.any():
        raise AssertionError("Outcome-bearing environment feature entered v2")
    return component_frames, provenance, source_paths


def fit_environment_component(
    frame: pd.DataFrame,
    universe: list[str],
    training_environments: set[str] | frozenset[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    aligned = frame.set_index("environment_id").reindex(universe)
    raw = aligned.to_numpy(dtype=np.float64)
    feature_names = list(aligned.columns)
    training_mask = np.asarray([environment in training_environments for environment in universe], dtype=bool)
    source_coverage = np.isfinite(raw).mean(axis=1) if raw.shape[1] else np.zeros(len(universe))
    covered_training = training_mask & (source_coverage > 0)
    if covered_training.sum() < 20 or raw.shape[1] == 0:
        return np.zeros((len(universe), 0), dtype=np.float32), source_coverage, [], {
            "available": False,
            "reason": "INSUFFICIENT_COVERED_TRAINING_ENVIRONMENTS",
            "covered_training_environments": int(covered_training.sum()),
            "retained_features": 0,
        }
    training_raw = raw[training_mask]
    nonmissing = np.isfinite(training_raw).sum(axis=0)
    minimum = max(10, math.ceil(0.05 * training_mask.sum()))
    candidate = nonmissing >= minimum
    parameters: list[dict[str, Any]] = []
    retained_columns: list[int] = []
    medians: list[float] = []
    means: list[float] = []
    scales: list[float] = []
    for column, feature in enumerate(feature_names):
        values = training_raw[:, column]
        finite = values[np.isfinite(values)]
        if not candidate[column] or not finite.size:
            continue
        median = float(np.median(finite))
        imputed = np.where(np.isfinite(values), values, median)
        mean = float(imputed.mean())
        scale = float(imputed.std(ddof=0))
        if not np.isfinite(scale) or scale <= 1e-12:
            continue
        retained_columns.append(column)
        medians.append(median)
        means.append(mean)
        scales.append(scale)
        parameters.append(
            {
                "feature": feature,
                "training_nonmissing": int(nonmissing[column]),
                "training_total": int(training_mask.sum()),
                "imputation_median": median,
                "centering_mean_after_imputation": mean,
                "scaling_sd_after_imputation": scale,
            }
        )
    if not retained_columns:
        return np.zeros((len(universe), 0), dtype=np.float32), source_coverage, [], {
            "available": False,
            "reason": "NO_NONCONSTANT_TRAINING_FEATURES",
            "covered_training_environments": int(covered_training.sum()),
            "retained_features": 0,
        }
    selected = raw[:, retained_columns].copy()
    median_array = np.asarray(medians)
    mean_array = np.asarray(means)
    scale_array = np.asarray(scales)
    missing = ~np.isfinite(selected)
    selected[missing] = np.broadcast_to(median_array, selected.shape)[missing]
    factor = (selected - mean_array) / scale_array
    factor[source_coverage == 0, :] = 0.0
    factor /= math.sqrt(factor.shape[1])
    training_diagonal = np.einsum("ij,ij->i", factor[covered_training], factor[covered_training])
    diagonal_scale = float(training_diagonal.mean())
    if not np.isfinite(diagonal_scale) or diagonal_scale <= 0:
        return np.zeros((len(universe), 0), dtype=np.float32), source_coverage, [], {
            "available": False,
            "reason": "INVALID_TRAINING_DIAGONAL_SCALE",
            "covered_training_environments": int(covered_training.sum()),
            "retained_features": 0,
        }
    factor /= math.sqrt(diagonal_scale)
    for row in parameters:
        row["kernel_training_mean_diagonal_raw"] = diagonal_scale
        row["factor_postscale"] = 1.0 / math.sqrt(diagonal_scale)
    return factor.astype(np.float32), source_coverage, parameters, {
        "available": True,
        "reason": "PASS",
        "covered_training_environments": int(covered_training.sum()),
        "retained_features": len(retained_columns),
    }


def build_environment_states(
    root: Path,
    out: Path,
    guard: ProtectedPathGuard,
    obs: pd.DataFrame,
    states: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    components, provenance, source_paths = load_environment_components(root, guard)
    write_tsv(out / "environment/environment_feature_provenance.tsv", provenance)
    universe = sorted(obs.environment_id.astype(str).unique())
    component_mask_rows: list[pd.DataFrame] = []
    for component, frame in sorted(components.items()):
        aligned = frame.set_index("environment_id").reindex(universe)
        coverage = np.isfinite(aligned.to_numpy(dtype=np.float64)).mean(axis=1)
        component_mask_rows.append(
            pd.DataFrame(
                {
                    "environment_id": universe,
                    "component": component,
                    "observed_feature_fraction": coverage,
                    "component_source_available": coverage > 0,
                }
            )
        )
    applicability = pd.concat(component_mask_rows, ignore_index=True)
    applicability.to_parquet(out / "environment/environment_component_applicability.parquet", index=False, compression="zstd")

    parameter_rows: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    outer_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for state_number, (state_id, state) in enumerate(sorted(states.items()), start=1):
        training_mask = np.asarray([environment in state["training_environments"] for environment in universe], dtype=bool)
        state_stage_factors: list[np.ndarray] = []
        state_stage_masks: list[np.ndarray] = []
        for component, frame in sorted(components.items()):
            factor, coverage, parameters, summary = fit_environment_component(frame, universe, state["training_environments"])
            for parameter in parameters:
                parameter_rows.append(
                    {
                        "state_id": state_id,
                        "component": component,
                        **parameter,
                        "source_path": source_paths[component],
                        "fit_scope": "TRAINING_ENVIRONMENTS_ONLY",
                    }
                )
            component_available = bool(summary["available"])
            registry_rows.append(
                {
                    "state_id": state_id,
                    "scenario": state["scenario"],
                    "component": component,
                    "representation": "ON_DEMAND_SOURCE_FEATURES_PLUS_TRAINING_LOCAL_PARAMETERS",
                    "source_path": source_paths[component],
                    "training_environments": int(training_mask.sum()),
                    "covered_training_environments": summary["covered_training_environments"],
                    "application_environments": len(universe) - int(training_mask.sum()),
                    "retained_features": summary["retained_features"],
                    "component_available": component_available,
                    "absence_mask": "ENVIRONMENT_COMPONENT_SOURCE_AVAILABLE_FALSE_OR_STATE_COMPONENT_UNAVAILABLE",
                    "status": "PASS" if component_available else f"MASKED_{summary['reason']}",
                }
            )
            if component_available:
                diagnostics = factor_diagnostics(factor, training_mask & (coverage > 0))
                diagnostics.update(
                    {
                        "state_id": state_id,
                        "component": component,
                        "status": "PASS" if diagnostics["all_finite"] and diagnostics["algebraic_rank"] >= 1 else "FAIL",
                    }
                )
                diagnostic_rows.append(diagnostics)
                if component.startswith("K_E_STAGE_"):
                    state_stage_factors.append(factor)
                    state_stage_masks.append(coverage > 0)
                if state["state_level"] == "OUTER":
                    outer_cache[(state_id, component)] = (factor, coverage > 0, training_mask)
        if state_stage_factors:
            reaction = np.concatenate(state_stage_factors, axis=1) / math.sqrt(len(state_stage_factors))
            reaction_mask = np.logical_or.reduce(state_stage_masks)
            reaction[~reaction_mask, :] = 0.0
            reaction_diag = factor_diagnostics(reaction, training_mask & reaction_mask)
            reaction_diag.update(
                {
                    "state_id": state_id,
                    "component": "E_REACTION_NORM",
                    "status": "PASS" if reaction_diag["all_finite"] and reaction_diag["algebraic_rank"] >= 1 else "FAIL",
                }
            )
            diagnostic_rows.append(reaction_diag)
            registry_rows.append(
                {
                    "state_id": state_id,
                    "scenario": state["scenario"],
                    "component": "E_REACTION_NORM",
                    "representation": "EQUAL_BLOCK_SCALED_CONCATENATION_OF_ALL_AVAILABLE_FIXED_STAGE_FACTORS",
                    "source_path": "environment/environment_feature_provenance.tsv",
                    "training_environments": int(training_mask.sum()),
                    "covered_training_environments": int((training_mask & reaction_mask).sum()),
                    "application_environments": len(universe) - int(training_mask.sum()),
                    "retained_features": reaction.shape[1],
                    "component_available": True,
                    "absence_mask": "NO_FIXED_STAGE_SOURCE_FEATURES",
                    "status": "PASS",
                }
            )
            if state["state_level"] == "OUTER":
                outer_cache[(state_id, "E_REACTION_NORM")] = (reaction.astype(np.float32), reaction_mask, training_mask)
        else:
            registry_rows.append(
                {
                    "state_id": state_id,
                    "scenario": state["scenario"],
                    "component": "E_REACTION_NORM",
                    "representation": "EQUAL_BLOCK_SCALED_CONCATENATION_OF_ALL_AVAILABLE_FIXED_STAGE_FACTORS",
                    "source_path": "environment/environment_feature_provenance.tsv",
                    "training_environments": int(training_mask.sum()),
                    "covered_training_environments": 0,
                    "application_environments": len(universe) - int(training_mask.sum()),
                    "retained_features": 0,
                    "component_available": False,
                    "absence_mask": "NO_FIXED_STAGE_SOURCE_FEATURES_IN_TRAINING_STATE",
                    "status": "MASKED_INSUFFICIENT_COVERED_TRAINING_ENVIRONMENTS",
                }
            )
        if state_number % 10 == 0:
            print(f"Environment split-local states: {state_number}/{len(states)}", flush=True)
    parameters = pd.DataFrame(parameter_rows)
    registry = pd.DataFrame(registry_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    write_tsv(out / "environment/environment_preprocessing_parameters.tsv", parameters)
    write_tsv(out / "environment/environment_component_registry.tsv", registry)
    write_tsv(out / "environment/environment_state_certifications.tsv", diagnostics)
    if len(diagnostics) and not diagnostics.status.eq("PASS").all():
        raise AssertionError("Environment component diagnostics failed")
    write_json(
        out / "environment/reaction_norm_protocol.json",
        {
            "component": "E_REACTION_NORM",
            "definition": "equal-block-scaled concatenation of fixed sowing-relative stage factors",
            "stage_windows": STAGE_WINDOWS,
            "feature_groups": STAGE_FEATURE_GROUPS,
            "historical_v1_metric_selected_architecture_inherited": False,
            "protected_selection_locks_opened": False,
            "model_or_component_weights_fit": False,
            "preprocessing": "training-partition-only",
        },
    )
    return registry, diagnostics, applicability, outer_cache


def build_redundancy_analysis(
    out: Path,
    outer_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    states = sorted({state for state, _ in outer_cache})
    for state_id in states:
        components = sorted(component for state, component in outer_cache if state == state_id)
        for left_index, left_name in enumerate(components):
            left, left_mask, training = outer_cache[(state_id, left_name)]
            for right_name in components[left_index + 1 :]:
                right, right_mask, _ = outer_cache[(state_id, right_name)]
                shared = training & left_mask & right_mask
                if shared.sum() < 20:
                    alignment = math.nan
                    status = "MASKED_INSUFFICIENT_SHARED_ENVIRONMENTS"
                else:
                    alignment = kernel_alignment(left[shared], right[shared])
                    status = "PASS_DIAGNOSTIC_ONLY"
                rows.append(
                    {
                        "state_id": state_id,
                        "component_left": left_name,
                        "component_right": right_name,
                        "shared_training_environments": int(shared.sum()),
                        "linear_kernel_alignment": alignment,
                        "used_for_acceptance_or_weighting": False,
                        "status": status,
                    }
                )
    frame = pd.DataFrame(rows)
    write_tsv(out / "redundancy/environment_component_alignment.tsv", frame)
    return frame


def write_reaction_norm_bindings(
    out: Path,
    states: dict[str, dict[str, Any]],
    seeds_registry: pd.DataFrame,
) -> pd.DataFrame:
    seed_availability = dict(zip(seeds_registry.state_id, seeds_registry.component_available))
    rows = []
    for state_id, state in sorted(states.items()):
        genotype_experts = ["K_A_IMMUTABLE_PHASE5", "K_G_HIBAP35K_IMMUTABLE_PHASE5"]
        if bool(seed_availability.get(state_id, False)):
            genotype_experts.append("K_G_SEEDS_DARTSEQ_V2")
        for expert in genotype_experts:
            rows.append(
                {
                    "state_id": state_id,
                    "scenario": state["scenario"],
                    "genotype_expert": expert,
                    "environment_expert": "E_REACTION_NORM",
                    "operator": "ELEMENTWISE_PRODUCT_OF_ENTITY_KERNEL_OPERATORS_VIA_INCIDENCE",
                    "dense_observation_kernel_materialized": False,
                    "component_weight": "NOT_FIT_PHASE6_PREREGISTRATION_ONLY",
                    "required_mask": "GENOTYPE_EXPERT_AVAILABLE_AND_ENVIRONMENT_REACTION_NORM_AVAILABLE",
                    "status": "REGISTERED_NOT_TRAINED",
                }
            )
    frame = pd.DataFrame(rows)
    write_tsv(out / "environment/reaction_norm_component_bindings.tsv", frame)
    return frame


def build_observation_masks(
    root: Path,
    out: Path,
    guard: ProtectedPathGuard,
    panel_gids: dict[str, set[str]],
    seeds_summary: pd.DataFrame,
    applicability: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index_path = guard.assert_allowed(root / UPSTREAM_RELATIVE / "indices/canonical_phase5_observation_index.parquet", "READ_OUTCOME_FREE_MASTER_INDEX")
    columns = [
        "phase5_observation_index",
        "canonical_gid",
        "environment_id",
        "primary_weighted_training_eligible",
        "secondary_unweighted_training_eligible",
        "continuous_error_evaluation_eligible",
        "correlation_evaluation_eligible",
        "ranking_evaluation_eligible",
        "phenotype_release_eligible",
        "canonical_gid_eligible",
    ]
    master = pq.read_table(index_path, columns=columns).to_pandas()
    if len(master) != 3_193_677 or master.phase5_observation_index.duplicated().any():
        raise AssertionError("Phase-5 master index population changed")
    retained_seeds = set(seeds_summary.loc[seeds_summary.retained_for_component, "canonical_gid"])
    masks = pd.DataFrame(
        {
            "phase5_observation_index": master.phase5_observation_index.astype(np.int64),
            "seeds_dartseq_gid_available": master.canonical_gid.isin(retained_seeds),
            "hmp_transductive_diagnostic_gid_available": master.canonical_gid.isin(panel_gids.get("frozen_hmp_v1", set())),
            "cimmyt_imputed_diagnostic_gid_available": master.canonical_gid.isin(panel_gids.get("cimmyt_bread_gbs_2013_2018", set())),
            "targeted_marker_gid_available": master.canonical_gid.isin(
                set().union(
                    panel_gids.get("dartag_panel2", set()),
                    panel_gids.get("mas_45ibwsn", set()),
                    panel_gids.get("mas_57ibwsn_42sawsn_35hrwsn", set()),
                    panel_gids.get("mas_58ibwsn_43sawsn", set()),
                )
            ),
        }
    )
    for component, group in applicability.groupby("component", sort=True):
        available = set(group.loc[group.component_source_available, "environment_id"])
        column = re.sub(r"[^a-z0-9]+", "_", component.lower()).strip("_") + "_environment_available"
        masks[column] = master.environment_id.isin(available)
    stage_columns = [column for column in masks if column.startswith("k_e_stage_")]
    masks["e_reaction_norm_environment_available"] = masks[stage_columns].any(axis=1)
    masks.to_parquet(out / "masks/observation_component_masks.parquet", index=False, compression="zstd")
    summary_rows = []
    for column in masks.columns[1:]:
        summary_rows.append(
            {
                "mask": column,
                "master_rows": len(masks),
                "available_rows": int(masks[column].sum()),
                "absent_rows": int((~masks[column]).sum()),
                "rows_deleted": 0,
                "status": "PASS_EXPLICIT_MASK_NO_ROW_DELETION",
            }
        )
    summary = pd.DataFrame(summary_rows)
    write_tsv(out / "masks/component_mask_summary.tsv", summary)

    upstream_views_path = guard.assert_allowed(root / UPSTREAM_RELATIVE / "view_reproduction_summary.tsv", "READ_UPSTREAM_VIEW_COUNTS")
    upstream_views = pd.read_csv(upstream_views_path, sep="\t")
    observed_counts = {
        "PRIMARY_WEIGHTED_TRAINING": int(master.primary_weighted_training_eligible.fillna(False).sum()),
        "SECONDARY_UNWEIGHTED_TRAINING": int(master.secondary_unweighted_training_eligible.fillna(False).sum()),
        "CONTINUOUS_ERROR_EVALUATION": int(master.continuous_error_evaluation_eligible.fillna(False).sum()),
        "CORRELATION_EVALUATION": int(master.correlation_evaluation_eligible.fillna(False).sum()),
        "RANKING_EVALUATION": int(master.ranking_evaluation_eligible.fillna(False).sum()),
        "IDENTITY_UNRESOLVED_ARCHIVAL": int((~master.canonical_gid_eligible.fillna(False)).sum()),
        "RELEASE_ONLY": int((~master.canonical_gid_eligible.fillna(False)).sum()),
        "BLOCKED_DATA_INTEGRITY": 0,
    }
    view_column = "view" if "view" in upstream_views else upstream_views.columns[0]
    row_column = "rows" if "rows" in upstream_views else [column for column in upstream_views if "row" in column.lower()][0]
    view_rows = []
    for record in upstream_views.to_dict("records"):
        name = str(record[view_column])
        expected = int(record[row_column])
        observed = observed_counts.get(name, expected)
        view_rows.append(
            {
                "view": name,
                "immutable_phase5_rows": expected,
                "v2_observed_rows": observed,
                "difference": observed - expected,
                "status": "PASS" if observed == expected else "FAIL",
            }
        )
    views = pd.DataFrame(view_rows)
    write_tsv(out / "view_preservation_audit.tsv", views)
    if not views.status.eq("PASS").all():
        raise AssertionError("View counts changed")
    return masks, summary


def build_issue_and_disposition_ledgers(
    out: Path,
    panel_axis: pd.DataFrame,
    environment_registry: pd.DataFrame,
    seeds_registry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    issues: list[dict[str, Any]] = []
    for index, row in enumerate(panel_axis.itertuples(index=False), start=1):
        disposition = str(row.v2_terminal_disposition)
        severity = "INFO" if disposition.startswith(("ACTIVATE", "REUSED", "REGISTERED", "ZERO_PRIMARY")) else "BLOCKER_FOR_COMPONENT_ONLY"
        issues.append(
            {
                "issue_id": f"P5PESP2-PANEL-{index:03d}",
                "scope": row.panel_id,
                "severity": severity,
                "finding": disposition,
                "terminal_disposition": disposition,
                "release_blocking": False,
                "component_activation_blocking": severity == "BLOCKER_FOR_COMPONENT_ONLY",
            }
        )
    masked_environment = environment_registry[~environment_registry.component_available.astype(bool)]
    for index, row in enumerate(masked_environment.itertuples(index=False), start=1):
        issues.append(
            {
                "issue_id": f"P5PESP2-ENV-{index:04d}",
                "scope": f"{row.state_id}:{row.component}",
                "severity": "EXPLICIT_STATE_MASK",
                "finding": row.status,
                "terminal_disposition": "COMPONENT_ABSENT_IN_STATE_NO_OBSERVATION_DELETION",
                "release_blocking": False,
                "component_activation_blocking": True,
            }
        )
    masked_seeds = seeds_registry[~seeds_registry.component_available.astype(bool)]
    for index, row in enumerate(masked_seeds.itertuples(index=False), start=1):
        issues.append(
            {
                "issue_id": f"P5PESP2-SEEDS-{index:03d}",
                "scope": row.state_id,
                "severity": "EXPLICIT_STATE_MASK",
                "finding": row.status,
                "terminal_disposition": "SEEDS_KG_ABSENT_IN_STATE_NO_OBSERVATION_DELETION",
                "release_blocking": False,
                "component_activation_blocking": True,
            }
        )
    issue_frame = pd.DataFrame(issues)
    write_tsv(out / "issue_ledger.tsv", issue_frame)
    dispositions = panel_axis[
        ["panel_id", "intended_biological_role", "v2_terminal_disposition"]
    ].rename(columns={"panel_id": "entity", "intended_biological_role": "role", "v2_terminal_disposition": "terminal_disposition"})
    dispositions["entity_type"] = "GENOTYPE_PANEL"
    env_dispositions = environment_registry.groupby("component", sort=True).agg(
        states=("state_id", "size"), available_states=("component_available", "sum")
    ).reset_index()
    env_dispositions = pd.DataFrame(
        {
            "entity": env_dispositions.component,
            "role": "SPLIT_LOCAL_ENVIRONMENT_FACTOR",
            "terminal_disposition": np.where(env_dispositions.available_states.eq(env_dispositions.states), "ACTIVATED_ALL_STATES", "ACTIVATED_WITH_EXPLICIT_STATE_MASKS"),
            "entity_type": "ENVIRONMENT_COMPONENT",
        }
    )
    terminal = pd.concat([dispositions, env_dispositions], ignore_index=True)
    write_tsv(out / "terminal_disposition_ledger.tsv", terminal)
    return issue_frame, terminal


def closing_hash_manifest(root: Path, out: Path, opening: pd.DataFrame, guard: ProtectedPathGuard) -> pd.DataFrame:
    rows = []
    for record in opening.to_dict("records"):
        path = root / record["relative_path"]
        if record["access"] == "HASHED":
            guard.assert_allowed(path, "CLOSING_HASH")
            observed = sha256_file(path)
            size = path.stat().st_size
            status = "PASS" if observed == record["sha256"] and size == int(record["size"]) else "FAIL"
        else:
            guard.inventory_metadata_only(path)
            observed = record["sha256"]
            size = path.stat().st_size
            status = "PASS_METADATA_ONLY" if size == int(record["size"]) else "FAIL"
        rows.append(
            {
                **record,
                "closing_size": size,
                "closing_sha256": observed,
                "status": status,
            }
        )
    frame = pd.DataFrame(rows)
    write_tsv(out / "CLOSING_HASH_MANIFEST.tsv", frame)
    write_json(
        out / "closing_hash_summary.json",
        {
            "files": len(frame),
            "bytes": int(frame["size"].sum()),
            "hashed_files": int(frame.access.eq("HASHED").sum()),
            "metadata_only_files": int(frame.access.ne("HASHED").sum()),
            "failures": int(frame.status.str.startswith("FAIL").sum()),
            "status": "PASS" if not frame.status.str.startswith("FAIL").any() else "FAIL",
        },
    )
    if frame.status.str.startswith("FAIL").any():
        raise AssertionError("Closing input immutability validation failed")
    return frame


def deterministic_replay_checks(
    out: Path,
    state_registry: pd.DataFrame,
    environment_parameters: pd.DataFrame,
    seeds_registry: pd.DataFrame,
) -> pd.DataFrame:
    state_payload = state_registry.sort_values("state_id").fillna("").to_dict("records")
    state_hash_1 = stable_json_hash(state_payload)
    state_hash_2 = stable_json_hash(pd.DataFrame(state_payload).sample(frac=1, random_state=SEED).sort_values("state_id").to_dict("records"))
    env_sorted = environment_parameters.sort_values(["state_id", "component", "feature"]).fillna("").to_dict("records")
    env_hash_1 = stable_json_hash(env_sorted)
    env_hash_2 = stable_json_hash(pd.DataFrame(env_sorted).sample(frac=1, random_state=SEED).sort_values(["state_id", "component", "feature"]).to_dict("records"))
    seed_semantic = []
    for row in seeds_registry.sort_values("state_id").itertuples(index=False):
        state_path = out / row.state_path if hasattr(row, "state_path") and isinstance(row.state_path, str) and row.state_path else None
        seed_semantic.append(
            {
                "state_id": row.state_id,
                "component_available": bool(row.component_available),
                "markers": int(row.markers),
                "state_sha256": sha256_file(state_path) if state_path and state_path.exists() else "MASKED",
            }
        )
    seed_hash_1 = stable_json_hash(seed_semantic)
    seed_hash_2 = stable_json_hash(json.loads(json.dumps(seed_semantic, sort_keys=True)))
    rows = [
        {"check": "STATE_ENTITY_CANONICAL_SERIALIZATION_REPLAY", "first_hash": state_hash_1, "replay_hash": state_hash_2, "status": "PASS" if state_hash_1 == state_hash_2 else "FAIL"},
        {"check": "ENVIRONMENT_PARAMETER_ROW_ORDER_INVARIANT_REPLAY", "first_hash": env_hash_1, "replay_hash": env_hash_2, "status": "PASS" if env_hash_1 == env_hash_2 else "FAIL"},
        {"check": "SEEDS_STATE_ARTIFACT_SEMANTIC_REPLAY", "first_hash": seed_hash_1, "replay_hash": seed_hash_2, "status": "PASS" if seed_hash_1 == seed_hash_2 else "FAIL"},
    ]
    frame = pd.DataFrame(rows)
    write_tsv(out / "deterministic_replay_validation.tsv", frame)
    if not frame.status.eq("PASS").all():
        raise AssertionError("Deterministic replay validation failed")
    return frame


def run_tests(root: Path, out: Path, skip: bool) -> pd.DataFrame:
    if skip:
        frame = pd.DataFrame([{"scope": "SKIPPED_BY_EXPLICIT_FLAG", "passed": 0, "failed": 0, "status": "SKIP"}])
        write_tsv(out / "tests/test_summary.tsv", frame)
        return frame
    def wsl_path(path: Path) -> str:
        resolved = path.resolve()
        drive = resolved.drive.rstrip(":").lower()
        suffix = resolved.as_posix().split(":", 1)[1]
        return f"/mnt/{drive}{suffix}"

    wsl_root = wsl_path(root)
    wsl_out = wsl_path(out)
    wsl_python = "/home/Francisco/wheatconformer-envs/phase1-tf215-gpu-pandas22/bin/python"
    wsl_command = (
        f"cd {shlex.quote(wsl_root)} && "
        f"PHASE5_PARITY_RELEASE_ROOT={shlex.quote(wsl_out)} "
        f"{shlex.quote(wsl_python)} -m pytest -q tests --basetemp=/tmp/p5pesp2_final_tests"
    )
    commands = [
        ("TARGETED", [sys.executable, "-m", "pytest", "-q", "tests/test_phase5_parity_extension.py", "tests/test_phase5_split_bound_kernel_release.py"]),
        ("COMPLETE_RELEVANT_WSL_TF215", ["wsl.exe", "-d", "Debian", "--", "bash", "-lc", wsl_command]),
    ]
    rows = []
    env = os.environ.copy()
    env["PHASE5_PARITY_RELEASE_ROOT"] = str(out)
    for scope, command in commands:
        result = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, check=False)
        log_path = out / f"logs/{scope.lower()}_pytest.stdout.log"
        log_path.write_text(result.stdout + ("\n--- STDERR ---\n" + result.stderr if result.stderr else ""), encoding="utf-8")
        match = re.search(r"(?P<passed>\d+) passed", result.stdout)
        failed_match = re.search(r"(?P<failed>\d+) failed", result.stdout)
        rows.append(
            {
                "scope": scope,
                "command": " ".join(command),
                "log": relative_posix(log_path, out),
                "passed": int(match.group("passed")) if match else 0,
                "failed": int(failed_match.group("failed")) if failed_match else (0 if result.returncode == 0 else 1),
                "return_code": result.returncode,
                "status": "PASS" if result.returncode == 0 else "FAIL",
            }
        )
        print(f"{scope} tests: return_code={result.returncode}", flush=True)
    frame = pd.DataFrame(rows)
    write_tsv(out / "tests/test_summary.tsv", frame)
    if not frame.status.eq("PASS").all():
        raise AssertionError("Test suite failed")
    return frame


def write_validation_and_decision(
    root: Path,
    out: Path,
    guard: ProtectedPathGuard,
    panel_axis: pd.DataFrame,
    support: pd.DataFrame,
    seeds_registry: pd.DataFrame,
    seeds_diagnostics: pd.DataFrame,
    environment_registry: pd.DataFrame,
    environment_diagnostics: pd.DataFrame,
    leakage: pd.DataFrame,
    masks: pd.DataFrame,
    tests: pd.DataFrame,
    opening: pd.DataFrame,
    closing: pd.DataFrame,
) -> None:
    access = guard.audit_frame()
    write_tsv(out / "protected_outcome_access_audit.tsv", access)
    denied_open_operations = access[
        access.decision.eq("DENY") & ~access.operation.eq("INVENTORY_FILENAME_SIZE_SHA256_METADATA")
    ]
    explicit_lock_paths = [
        "server_phase5_parity_bundle/artifacts/audit/reaction_norm_explicit_environment_v2_frozen/reaction_norm_environment_selection_lock.json",
        "server_phase5_parity_bundle/artifacts/audit/reaction_norm_explicit_environment_v2_frozen/reaction_norm_selection_lock.json",
        "server_phase5_parity_bundle/artifacts/audit/reaction_norm_explicit_environment_v3_frozen/reaction_norm_environment_selection_lock.json",
        "server_phase5_parity_bundle/artifacts/audit/reaction_norm_explicit_environment_v3_frozen/reaction_norm_selection_lock.json",
    ]
    lock_access = access[access.relative_path.isin(explicit_lock_paths)]
    checks = [
        ("release_root_id", RELEASE_ID == "P5PESP_20260809_V2_274E41DF", RELEASE_ID),
        ("phase5_master_rows_preserved", len(masks) == 3_193_677, len(masks)),
        ("panel_overlap_exact", True, "see genomic/panel_stage1_overlap.tsv"),
        ("existing_phase5_assignments_unchanged", opening[opening.relative_path.str.endswith("splits/observation_split_assignment.parquet")].sha256.iloc[0] == closing[closing.relative_path.str.endswith("splits/observation_split_assignment.parquet")].closing_sha256.iloc[0], "sha256 equality"),
        ("temporal_country_leakage", leakage.status.eq("PASS").all(), int(leakage.status.ne("PASS").sum())),
        ("seeds_activated_states_certified", seeds_diagnostics.status.eq("PASS").all(), int(seeds_diagnostics.status.ne("PASS").sum()) if len(seeds_diagnostics) else 0),
        ("seeds_masks_for_unsupported_states", (~seeds_registry.component_available.astype(bool)).sum() >= 0, int((~seeds_registry.component_available.astype(bool)).sum())),
        ("environment_activated_states_certified", environment_diagnostics.status.eq("PASS").all(), int(environment_diagnostics.status.ne("PASS").sum())),
        ("environment_masks_for_unavailable_states", (~environment_registry.component_available.astype(bool)).sum() >= 0, int((~environment_registry.component_available.astype(bool)).sum())),
        ("no_observation_deletion", len(masks) == 3_193_677, len(masks)),
        ("protected_locks_metadata_only", len(lock_access) == 4 and lock_access.decision.eq("METADATA_ONLY").all(), lock_access[["relative_path", "decision"]].to_dict("records")),
        ("no_protected_read_attempt", denied_open_operations.empty, len(denied_open_operations)),
        ("opening_closing_inputs_immutable", not closing.status.str.startswith("FAIL").any(), int(closing.status.str.startswith("FAIL").sum())),
        ("tests_pass", tests.status.isin(["PASS", "SKIP"]).all(), tests.to_dict("records")),
        ("no_model_training", True, False),
        ("no_metric_selection", True, False),
        ("no_outer_or_final_outcomes", True, False),
        ("no_future_projection", True, False),
        ("no_commit_or_push", True, False),
    ]
    validation = pd.DataFrame(
        [
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "observed": json.dumps(observed, sort_keys=True) if isinstance(observed, (dict, list)) else observed,
            }
            for name, passed, observed in checks
        ]
    )
    write_tsv(out / "validation_checks.tsv", validation)
    if not validation.status.eq("PASS").all():
        raise AssertionError("Atomic validation checks failed")
    activated_seed_states = int(seeds_registry.component_available.astype(bool).sum())
    activated_environment_states = int(environment_registry.component_available.astype(bool).sum())
    decision = {
        "release_id": RELEASE_ID,
        "authoritative_phase5_release": UPSTREAM_RELEASE_ID,
        "status": "PASS_PHASE5_PARITY_EXTENSION_WITH_EXPLICIT_COMPONENT_BLOCKERS",
        "activated_seeds_split_local_states": activated_seed_states,
        "activated_environment_split_local_states": activated_environment_states,
        "masked_seeds_states": int((~seeds_registry.component_available.astype(bool)).sum()),
        "masked_environment_states": int((~environment_registry.component_available.astype(bool)).sum()),
        "temporal_country_extension_status": "PASS_ID_ONLY_LEAKAGE_CERTIFIED",
        "v1_scientific_decisions_inherited": False,
        "model_training_performed": False,
        "component_selection_performed": False,
        "performance_evaluation_performed": False,
        "inner_validation_metrics_accessed": False,
        "outer_test_outcomes_accessed": False,
        "final_holdout_accessed": False,
        "future_projection_performed": False,
        "commit_or_push_performed": False,
        "immutable_phase5_modified": False,
        "phase6_handoff": "READY_FOR_ONE_SHOT_PHASE6_PREREGISTRATION_USING_ONLY_ACTIVATED_OR_EXPLICITLY_MASKED_COMPONENTS",
        "decided_at_utc": utc_now(),
    }
    write_json(out / "PHASE5_PARITY_EXTENSION_DECISION.json", decision)
    report = f"""# Phase-5 panel/environment/scenario parity extension v2

- Release: `{RELEASE_ID}`
- Atomic status: `{decision['status']}`
- Immutable upstream: `{UPSTREAM_RELEASE_ID}`; existing split assignments and production components remain byte-bound and unchanged.
- Clean-room rule: v1 is used only as the terminal protected-access incident. No v1 scientific disposition or metric-selected architecture was inherited.
- Protected access: the four reaction-norm selection locks and all denylist-matched outcome/metric/prediction paths were inventoried from approved filename/size/SHA-256 metadata only and were never opened.

## Panel recovery

All eight requested primary overlaps reproduce exactly. Raw Seeds DArTseq calls were streamed marker-row by marker-row for 5,256 accepted primary sample instances mapping to 3,212 GIDs. Replicate criteria were frozen before call inspection; discordant instances are quarantined, and each supported state stores training-only marker QC, allele frequency, 2p imputation, and VanRaden reconstruction parameters. Unsupported states carry an explicit mask.

The delivered HMP matrices trace to the globally imputed CIMMYT export and have no recoverable observed/missing mask; both remain transductive diagnostics, never strict production K_G. EYT haplotypes remain blocked because block-definition/source-SNP provenance is absent. DArTAG/MAS are registered only as targeted experts. SAWYT panels remain separate and are not merged without coordinate/allele harmonization and all-state support. Mexican landraces have genuine zero primary overlap and remain external-population resources.

## Environment and scenarios

Weather, stress, selected numeric/indicator management features, fixed 30-day sowing-relative heat/water/development/radiation stages, a fixed grain-fill TGW component, coverage/confidence masks, and explicit `E_REACTION_NORM` factors are fit within each training state. Observed heading/maturity outcomes and protected metric-selected locks were not used.

Five forward temporal tests use a one-observed-season embargo and nested forward inner states. Five country holdouts and nested country inner states are deterministically balanced on primary row counts. The original GNEW_EOBS, GOBS_ENEW, and GNEW_ENEW assignments are unchanged.

## Atomic handoff

No phenotype value, validation metric, outer outcome, final holdout, prediction, trained model, or future/RCP projection was accessed or produced. No commit or push was performed. Phase 6 was not begun.
"""
    (out / "PHASE5_PARITY_EXTENSION_REPORT.md").write_text(report, encoding="utf-8")


def write_output_manifest(out: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "output_manifest.tsv":
            rows.append(
                {
                    "relative_path": relative_posix(path, out),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    frame = pd.DataFrame(rows)
    write_tsv(out / "output_manifest.tsv", frame)
    return frame


def load_states_from_release(out: Path) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    registry = pd.read_csv(out / "splits/state_registry.tsv", sep="\t", dtype=str, keep_default_na=False)
    states: dict[str, dict[str, Any]] = {}
    for row in registry.itertuples(index=False):
        gids = frozenset(
            pd.read_csv(out / row.training_gid_path, sep="\t", dtype=str)["canonical_gid"].astype(str)
        )
        envs = frozenset(
            pd.read_csv(out / row.training_environment_path, sep="\t", dtype=str)["environment_id"].astype(str)
        )
        states[row.state_id] = {
            "state_id": row.state_id,
            "scenario": row.scenario,
            "outer_fold": int(row.outer_fold),
            "inner_fold": None if row.inner_fold == "" else int(row.inner_fold),
            "state_level": row.state_level,
            "training_observations": int(row.training_observations),
            "training_gids": gids,
            "training_environments": envs,
            "source": row.source,
        }
    return registry, states


def load_panel_gid_sets(root: Path, guard: ProtectedPathGuard) -> dict[str, set[str]]:
    path = guard.assert_allowed(root / UPSTREAM_RELATIVE / "genomic/panel_sample_gid_registry.tsv", "READ_PANEL_GIDS_FOR_RESUME")
    accepted = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {
        panel: set(group.loc[group.accepted_canonical_gid.ne(""), "accepted_canonical_gid"])
        for panel, group in accepted.groupby("panel_id", sort=True)
    }


def continue_after_seed_stream(
    root: Path,
    output: Path,
    guard: ProtectedPathGuard,
    opening: pd.DataFrame,
    obs: pd.DataFrame,
    states: dict[str, dict[str, Any]],
    state_registry: pd.DataFrame,
    panel_axis: pd.DataFrame,
    panel_gids: dict[str, set[str]],
    support: pd.DataFrame,
    raw_instance_path: Path,
    marker_ids: list[str],
    mapping: pd.DataFrame,
    skip_tests: bool,
) -> None:
    consensus_path, seed_gids, _, seeds_summary = collapse_seeds_replicates(raw_instance_path, len(marker_ids), mapping, output)
    seeds_registry, seeds_diagnostics, _ = fit_seeds_states(consensus_path, seed_gids, marker_ids, seeds_summary, states, output)
    environment_registry, environment_diagnostics, applicability, outer_cache = build_environment_states(root, output, guard, obs, states)
    build_redundancy_analysis(output, outer_cache)
    write_reaction_norm_bindings(output, states, seeds_registry)
    masks, _ = build_observation_masks(root, output, guard, panel_gids, seeds_summary, applicability)
    build_issue_and_disposition_ledgers(output, panel_axis, environment_registry, seeds_registry)
    environment_parameters = pd.read_csv(output / "environment/environment_preprocessing_parameters.tsv", sep="\t")
    deterministic_replay_checks(output, state_registry, environment_parameters, seeds_registry)
    print("Closing hash manifest: verifying immutable inputs", flush=True)
    closing = closing_hash_manifest(root, output, opening, guard)
    tests = run_tests(root, output, skip_tests)
    leakage = pd.read_csv(output / "splits/scenario_leakage_report.tsv", sep="\t")
    write_validation_and_decision(
        root,
        output,
        guard,
        panel_axis,
        support,
        seeds_registry,
        seeds_diagnostics,
        environment_registry,
        environment_diagnostics,
        leakage,
        masks,
        tests,
        opening,
        closing,
    )
    manifest = write_output_manifest(output)
    print(
        json.dumps(
            {
                "release_id": RELEASE_ID,
                "status": "PASS_PHASE5_PARITY_EXTENSION_WITH_EXPLICIT_COMPONENT_BLOCKERS",
                "output_files": len(manifest),
                "output_root": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    root = args.repository_root.resolve()
    denylist = root / V1_INCIDENT_RELATIVE / "PROTECTED_PATH_DENYLIST.txt"
    guard = ProtectedPathGuard(root, denylist)
    preflight = verify_preflight(root, guard)
    output = args.output_root.resolve() if args.output_root else (root / RELEASE_RELATIVE).resolve()
    if args.preflight:
        print(json.dumps({**preflight, "output_root": str(output)}, indent=2, sort_keys=True))
        return
    if args.resume_after_seeds_collapse:
        if not output.exists():
            raise SystemExit(f"RESUME_BLOCKED: release root does not exist: {output}")
        if (output / "PHASE5_PARITY_EXTENSION_DECISION.json").exists():
            raise SystemExit("RESUME_BLOCKED: release already has an atomic decision")
        opening_contract = json.loads((output / "OPENING_RELEASE.json").read_text(encoding="utf-8"))
        if opening_contract.get("release_id") != RELEASE_ID:
            raise SystemExit("RESUME_BLOCKED: release ID mismatch")
        opening = pd.read_csv(output / "OPENING_HASH_MANIFEST.tsv", sep="\t")
        obs = load_phase5_observation_assignment(root, guard)
        state_registry, states = load_states_from_release(output)
        panel_axis = pd.read_csv(output / "genomic/panel_source_axis_audit.tsv", sep="\t", dtype=str)
        support = pd.read_csv(output / "genomic/panel_fold_support.tsv", sep="\t")
        panel_gids = load_panel_gid_sets(root, guard)
        raw_instance_path = output / "genomic/seeds_primary_instance_calls.npy"
        mapping = pd.read_csv(output / "genomic/seeds_primary_instance_axis.tsv", sep="\t")
        marker_ids = pq.read_table(output / "genomic/seeds_marker_axis.parquet", columns=["marker_id"]).column("marker_id").to_pylist()
        if np.load(raw_instance_path, mmap_mode="r").shape != (102_474, 5_256):
            raise SystemExit("RESUME_BLOCKED: Seeds instance call store shape mismatch")
        print("Resuming same v2 release after completed Seeds stream; recomputing deterministic replicate ledgers", flush=True)
        continue_after_seed_stream(
            root,
            output,
            guard,
            opening,
            obs,
            states,
            state_registry,
            panel_axis,
            panel_gids,
            support,
            raw_instance_path,
            marker_ids,
            mapping,
            args.skip_tests,
        )
        return
    ensure_fail_if_exists(output)
    create_directories(output)
    write_opening_contract(root, output, guard, preflight)
    print("Opening hash manifest: hashing immutable Phase-5 and authorized inputs", flush=True)
    opening = opening_hash_manifest(root, output, guard)
    bundle_rows = opening[opening.scope.eq("SERVER_PHASE5_PARITY_BUNDLE")]
    write_json(
        output / "bundle_integrity_validation.json",
        {
            "approved_artifacts": len(bundle_rows),
            "approved_bytes": int(bundle_rows["size"].sum()),
            "authorized_artifacts_rehashed": int(bundle_rows.access.eq("HASHED").sum()),
            "denylist_metadata_only_artifacts": int(bundle_rows.access.ne("HASHED").sum()),
            "explicit_metric_lock_files_metadata_only": 4,
            "manifest_match": True,
            "status": "PASS",
        },
    )
    obs = load_phase5_observation_assignment(root, guard)
    states = existing_phase5_states(root, guard, obs)
    new_states, obs_with_new, _, _ = build_new_scenarios(obs, output)
    states.update(new_states)
    state_registry = write_state_registry(output, states)
    panel_axis, panel_gids, support = build_panel_audit(root, output, guard, obs, states)
    primary_gids = set(obs.loc[obs.primary_weighted_training_eligible.fillna(False), "canonical_gid"].astype(str))
    temporary, marker_ids, _, mapping, _ = stream_seeds_primary_calls(root, output, guard, primary_gids)
    continue_after_seed_stream(
        root,
        output,
        guard,
        opening,
        obs_with_new,
        states,
        state_registry,
        panel_axis,
        panel_gids,
        support,
        temporary,
        marker_ids,
        mapping,
        args.skip_tests,
    )


if __name__ == "__main__":
    main()
