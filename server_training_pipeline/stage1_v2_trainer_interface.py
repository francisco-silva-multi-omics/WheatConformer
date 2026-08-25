from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse


PHASE5 = Path("audit/v2/phase5_split_bound_kernel_validation_v2")
PARITY = Path("audit/v2/phase5_panel_environment_scenario_parity_extension_v2")
KA_EXTENSION = Path("audit/v2/phase5_ka_temporal_country_extension_v1")
H_SEEDS = Path("audit/v2/phase6_h_seeds_operator_v1")
CIMMYT = Path("audit/v2/phase5_cimmyt_pre_qc_split_local_v1")
PROJECTION = Path("environment/v2/e_projection_core_v1_split_bound_historical_v1")
SELECTION_PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_selection_protocol_v1.json"
)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype("string").fillna("").str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "pass"}
    )


def normalize_cycle_year(value: object) -> str:
    """Return the Phase-5 normalized season-end year as a string identifier."""
    text = str(value).strip()
    if len(text) == 4 and text.isdigit():
        return text
    if (
        len(text) == 5
        and text[2] == "-"
        and text[:2].isdigit()
        and text[3:].isdigit()
    ):
        end = int(text[3:])
        return str(1900 + end if int(text[:2]) >= 70 else 2000 + end)
    raise ValueError(f"Unsupported cycle/year label: {value!r}")


def normalized_cycle_years(values: pd.Series) -> pd.Series:
    return values.map(normalize_cycle_year).astype("string")


def state_role_masks(
    observations: pd.DataFrame,
    *,
    scenario: str,
    outer_fold: int,
    inner_fold: int,
    training_gids: set[str],
    training_environments: set[str],
    assignments: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    role_column = f"{scenario.lower()}_outer{outer_fold}_role"
    outer_role = observations[role_column].astype("string").fillna("")
    outer_training = outer_role.eq("TRAIN")
    outer_test = outer_role.isin({"TEST", "OUTER_TEST_ID_ONLY"})
    gid_training = observations["canonical_gid"].astype(str).isin(training_gids)
    env_training = observations["environment_id"].astype(str).isin(
        training_environments
    )

    if scenario == "GNEW_EOBS":
        training = outer_training & gid_training & env_training
        validation = outer_training & ~gid_training & env_training
    elif scenario == "GOBS_ENEW":
        training = outer_training & gid_training & env_training
        validation = outer_training & gid_training & ~env_training
    elif scenario == "GNEW_ENEW":
        training = outer_training & gid_training & env_training
        validation = outer_training & ~gid_training & ~env_training
    else:
        if assignments is None:
            raise ValueError(f"{scenario} requires frozen inner entity assignments")
        local = assignments.loc[
            assignments["scenario"].eq(scenario)
            & assignments["outer_fold"].eq(str(outer_fold))
            & assignments["inner_fold"].eq(str(inner_fold))
        ]
        if scenario == "TEMPORAL_YEAR":
            values = normalized_cycle_years(observations["year"])
            entity_type = "NORMALIZED_YEAR"
        elif scenario == "COUNTRY_HOLDOUT":
            values = observations["country"].astype("string").fillna("")
            entity_type = "COUNTRY"
        else:
            raise ValueError(f"Unsupported scenario: {scenario}")
        mapping = local.loc[local["entity_type"].eq(entity_type)].set_index(
            "entity_id"
        )["assignment"]
        assigned = values.map(mapping).fillna("")
        training = outer_training & assigned.eq("TRAIN")
        validation = outer_training & assigned.eq("INNER_VALIDATION_ID_ONLY")

    embargo = ~(training | validation | outer_test)
    if bool((training & validation).any()) or bool((training & outer_test).any()):
        raise ValueError("Stage-1 v2 role masks overlap")
    return training, validation, embargo


@dataclass(frozen=True)
class SparsePedigreeSpec:
    factor_path: str
    mendelian_variance_path: str
    entity_order_path: str
    entity_count: int
    factor_nonzero_count: int
    training_scale: float
    state_hash: str


@dataclass(frozen=True)
class MarkerFactorSpec:
    component: str
    raw_dosage_path: str
    entity_axis_path: str
    parameter_path: str
    entity_count: int
    training_entity_count: int
    retained_marker_count: int
    denominator: float
    absence_mask: str
    state_hash: str


@dataclass(frozen=True)
class EnvironmentFactorSpec:
    component: str
    entity_axis_path: str
    standardized_features_path: str
    missing_mask_path: str
    kernel_factor_path: str
    kernel_projection_path: str
    environment_count: int
    active_environment_count: int
    inactive_environment_count: int
    feature_count: int
    factor_rank: int


@dataclass(frozen=True)
class HSeedsOperatorSpec:
    state_path: str
    overlap_entity_count: int
    training_overlap_count: int
    genomic_blend_weight: float
    component_available: bool
    absence_mask: str
    state_hash: str


@dataclass(frozen=True)
class Stage1V2StateSpec:
    state_id: str
    scenario: str
    outer_fold: int
    inner_fold: int | None
    state_level: str
    candidate: str
    training_observation_count: int
    validation_observation_count: int
    embargo_observation_count: int
    outer_test_identifier_count: int
    pedigree: SparsePedigreeSpec
    marker_factors: tuple[MarkerFactorSpec, ...]
    projection_environment: EnvironmentFactorSpec | None
    historical_environment_registry_path: str | None
    h_seeds: HSeedsOperatorSpec | None
    phenotype_values_read: bool = False
    outer_test_outcomes_read: bool = False
    final_holdout_outcomes_read: bool = False


def _one(frame: pd.DataFrame, key: str, value: str, label: str) -> pd.Series:
    selected = frame.loc[frame[key].astype(str).eq(value)]
    if len(selected) != 1:
        raise ValueError(f"Expected one {label} row for {value}; observed={len(selected)}")
    return selected.iloc[0]


def load_selection_protocol(root: Path) -> dict[str, Any]:
    code_root = Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
    path = code_root / SELECTION_PROTOCOL
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "stage1_v2_phase6_model_selection_v1":
        raise ValueError("Unexpected Stage-1 v2 selection protocol")
    return protocol


def _load_state_registry(root: Path) -> pd.DataFrame:
    registry = pd.read_csv(root / PARITY / "splits/state_registry.tsv", sep="\t", dtype=str)
    if len(registry) != 150 or not registry["state_id"].is_unique:
        raise ValueError("Stage-1 v2 trainer requires the exact 150-state registry")
    return registry


def _root_path(root: Path, relative: object) -> Path:
    path = (root / str(relative)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Artifact escapes repository root: {relative}") from exc
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _outer_role_column(scenario: str, outer_fold: int) -> str:
    return f"{scenario.lower()}_outer{outer_fold}_role"


def identifier_role_counts(root: Path, state: pd.Series) -> dict[str, int]:
    scenario = str(state["scenario"])
    outer_fold = int(state["outer_fold"])
    inner_text = "" if pd.isna(state["inner_fold"]) else str(state["inner_fold"]).strip()
    inner_fold = int(inner_text) if inner_text else None
    primary = root / PHASE5 / "splits/observation_split_assignment.parquet"
    base_columns = [
        "phase4_adjusted_row_id",
        "canonical_gid",
        "environment_id",
        "year",
        "country",
        "primary_weighted_training_eligible",
    ]
    if scenario in {"GNEW_EOBS", "GOBS_ENEW", "GNEW_ENEW"}:
        role_column = _outer_role_column(scenario, outer_fold)
        observations = pd.read_parquet(primary, columns=[*base_columns, role_column])
    else:
        observations = pd.read_parquet(primary, columns=base_columns)
        parity_roles = pd.read_parquet(
            root / PARITY / "splits/scenario_observation_roles.parquet",
            columns=["phase4_adjusted_row_id", _outer_role_column(scenario, outer_fold)],
        )
        role_column = _outer_role_column(scenario, outer_fold)
        observations = observations.merge(
            parity_roles, on="phase4_adjusted_row_id", how="left", validate="one_to_one"
        )
    eligible = bool_series(observations["primary_weighted_training_eligible"])
    observations = observations.loc[eligible].copy()
    outer_role = observations[role_column].astype("string").fillna("")
    outer_training = outer_role.eq("TRAIN")
    outer_test = outer_role.isin({"TEST", "OUTER_TEST_ID_ONLY"})
    if inner_fold is None:
        return {
            "training": int(outer_training.sum()),
            "validation": 0,
            "embargo": int((~outer_training & ~outer_test).sum()),
            "outer_test_identifiers": int(outer_test.sum()),
        }

    training_gids = set(
        pd.read_csv(root / PARITY / str(state["training_gid_path"]), sep="\t", dtype=str)[
            "canonical_gid"
        ].astype(str)
    )
    training_envs = set(
        pd.read_csv(
            root / PARITY / str(state["training_environment_path"]), sep="\t", dtype=str
        )["environment_id"].astype(str)
    )
    assignments = None
    if scenario in {"TEMPORAL_YEAR", "COUNTRY_HOLDOUT"}:
        assignments = pd.read_csv(
            root / PARITY / "splits/inner_entity_assignment.tsv", sep="\t", dtype=str
        )
    training, validation, embargo = state_role_masks(
        observations,
        scenario=scenario,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        training_gids=training_gids,
        training_environments=training_envs,
        assignments=assignments,
    )
    return {
        "training": int(training.sum()),
        "validation": int(validation.sum()),
        "embargo": int(embargo.sum()),
        "outer_test_identifiers": int(outer_test.sum()),
    }


def _pedigree_spec(root: Path, state_id: str) -> SparsePedigreeSpec:
    registry = pd.read_csv(
        root / KA_EXTENSION / "pedigree/combined_150_state_ka_registry.tsv",
        sep="\t",
        dtype=str,
    )
    row = _one(registry, "state_id", state_id, "K_A")
    factor_path = _root_path(root, row["raw_operator_factor_root_relative"])
    d_path = _root_path(root, row["raw_operator_d_root_relative"])
    entity_path = _root_path(root, row["entity_order_root_relative"])
    factor = sparse.load_npz(factor_path)
    d_values = np.load(d_path, mmap_mode="r")
    if factor.shape[0] != factor.shape[1] or factor.shape[0] != len(d_values):
        raise ValueError("K_A sparse factor dimensions disagree")
    entities = pd.read_csv(entity_path, sep="\t", dtype=str)
    return SparsePedigreeSpec(
        factor_path=factor_path.relative_to(root).as_posix(),
        mendelian_variance_path=d_path.relative_to(root).as_posix(),
        entity_order_path=entity_path.relative_to(root).as_posix(),
        entity_count=len(entities),
        factor_nonzero_count=int(factor.nnz),
        training_scale=float(row["training_scale_mean_diagonal"]),
        state_hash=str(row["state_hash"]),
    )


def _seeds_spec(root: Path, state_id: str) -> MarkerFactorSpec:
    registry = pd.read_csv(
        root / PARITY / "genomic/seeds_component_registry.tsv", sep="\t", dtype=str
    )
    row = _one(registry, "state_id", state_id, "Seeds")
    parameter_path = _root_path(root, PARITY / str(row["state_path"]))
    raw_path = _root_path(root, PARITY / str(row["raw_consensus_path"]))
    axis_path = _root_path(root, PARITY / "genomic/seeds_gid_consensus_summary.tsv")
    parameters = np.load(parameter_path, allow_pickle=False)
    return MarkerFactorSpec(
        component="K_G_SEEDS_DARTSEQ_V2",
        raw_dosage_path=raw_path.relative_to(root).as_posix(),
        entity_axis_path=axis_path.relative_to(root).as_posix(),
        parameter_path=parameter_path.relative_to(root).as_posix(),
        entity_count=int(row["entities"]),
        training_entity_count=int(row["training_entities"]),
        retained_marker_count=len(parameters["retained_marker_index"]),
        denominator=float(np.asarray(parameters["denominator"]).reshape(-1)[0]),
        absence_mask=str(row["absence_mask"]),
        state_hash=str(row["state_sha256"]),
    )


def _cimmyt_spec(root: Path, state_id: str) -> MarkerFactorSpec:
    registry = pd.read_csv(
        root / CIMMYT / "states/cimmyt_pre_qc_component_registry.tsv", sep="\t", dtype=str
    )
    row = _one(registry, "state_id", state_id, "CIMMYT pre-QC")
    preprocessing = pd.read_csv(
        root / CIMMYT / "states/cimmyt_pre_qc_fold_preprocessing_registry.tsv",
        sep="\t",
        dtype=str,
    )
    prep = _one(preprocessing, "state_id", state_id, "CIMMYT preprocessing")
    parameter_path = _root_path(root, CIMMYT / str(prep["parameter_path"]))
    raw_path = _root_path(root, CIMMYT / "genomic/cimmyt_pre_qc_primary_raw_calls.npy")
    axis_path = _root_path(root, CIMMYT / "genomic/cimmyt_pre_qc_primary_sample_axis.tsv")
    axis = pd.read_csv(axis_path, sep="\t", dtype=str)
    parameters = np.load(parameter_path, allow_pickle=False)
    denominator = float(np.asarray(parameters["vanraden_denominator"]).reshape(-1)[0])
    return MarkerFactorSpec(
        component="K_G_CIMMYT_PRE_QC",
        raw_dosage_path=raw_path.relative_to(root).as_posix(),
        entity_axis_path=axis_path.relative_to(root).as_posix(),
        parameter_path=parameter_path.relative_to(root).as_posix(),
        entity_count=len(axis),
        training_entity_count=int(row["training_entities"]),
        retained_marker_count=len(parameters["retained_marker_index"]),
        denominator=denominator,
        absence_mask="GID_NOT_IN_SOURCE_INTRINSIC_QC_PASSING_PANEL",
        state_hash=str(prep["parameter_sha256"]),
    )


def _projection_spec(root: Path, state_id: str) -> EnvironmentFactorSpec:
    state_dir = _root_path(root, PROJECTION / "states" / state_id)
    entity_path = state_dir / "environment_entities.tsv"
    features_path = state_dir / "standardized_features_float32.npy"
    mask_path = state_dir / "feature_missing_mask_packbits.npy"
    factor_path = state_dir / "kernel_factor_float32.npy"
    projection_path = state_dir / "kernel_projection_float32.npy"
    for path in (entity_path, features_path, mask_path, factor_path, projection_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    entities = pd.read_csv(entity_path, sep="\t", dtype=str)
    active = bool_series(entities["component_active"])
    features = np.load(features_path, mmap_mode="r")
    factor = np.load(factor_path, mmap_mode="r")
    projection = np.load(projection_path, mmap_mode="r")
    if features.shape != (len(entities), 153):
        raise ValueError("Projection feature matrix has an unexpected shape")
    if factor.shape[0] != len(entities) or projection.shape != (153, factor.shape[1]):
        raise ValueError("Projection factors have inconsistent dimensions")
    return EnvironmentFactorSpec(
        component="E_PROJECTION_CORE_V1",
        entity_axis_path=entity_path.relative_to(root).as_posix(),
        standardized_features_path=features_path.relative_to(root).as_posix(),
        missing_mask_path=mask_path.relative_to(root).as_posix(),
        kernel_factor_path=factor_path.relative_to(root).as_posix(),
        kernel_projection_path=projection_path.relative_to(root).as_posix(),
        environment_count=len(entities),
        active_environment_count=int(active.sum()),
        inactive_environment_count=int((~active).sum()),
        feature_count=features.shape[1],
        factor_rank=factor.shape[1],
    )


def _h_seeds_spec(root: Path, state_id: str) -> HSeedsOperatorSpec:
    registry = pd.read_csv(root / H_SEEDS / "h_seeds_operator_registry.tsv", sep="\t", dtype=str)
    row = _one(registry, "state_id", state_id, "H_SEEDS")
    state_path = _root_path(root, row["state_path"])
    if sha256_file(state_path) != str(row["state_sha256"]):
        raise ValueError(f"H_SEEDS state checksum mismatch: {state_id}")
    return HSeedsOperatorSpec(
        state_path=state_path.relative_to(root).as_posix(),
        overlap_entity_count=int(row["operator_overlap_gids"]),
        training_overlap_count=int(row["training_overlap_gids"]),
        genomic_blend_weight=float(row["genomic_blend_weight"]),
        component_available=bool(
            bool_series(pd.Series([row["component_available"]])).iloc[0]
        ),
        absence_mask=str(row["absence_mask"]),
        state_hash=str(row["state_hash"]),
    )


def load_state_spec(root: Path, state_id: str, candidate: str) -> Stage1V2StateSpec:
    root = root.resolve()
    protocol = load_selection_protocol(root)
    if candidate not in protocol["candidates"]:
        raise ValueError(f"Candidate is not preregistered: {candidate}")
    state = _one(_load_state_registry(root), "state_id", state_id, "state")
    requirements = protocol["candidates"][candidate]
    genotype_components = set(requirements["genotype_components"])
    environment_components = set(requirements["environment_components"])
    markers = []
    if "K_G_SEEDS_DARTSEQ_V2" in genotype_components:
        markers.append(_seeds_spec(root, state_id))
    if "K_G_CIMMYT_PRE_QC" in genotype_components:
        markers.append(_cimmyt_spec(root, state_id))
    h_spec = _h_seeds_spec(root, state_id) if "H_SEEDS" in genotype_components else None
    projection = (
        _projection_spec(root, state_id)
        if "E_PROJECTION_CORE_V1" in environment_components
        else None
    )
    historical_registry = None
    if "HISTORICAL_PARITY_COMPONENTS" in environment_components:
        path = _root_path(root, PARITY / "environment/environment_component_registry.tsv")
        historical_registry = path.relative_to(root).as_posix()
    counts = identifier_role_counts(root, state)
    inner_text = "" if pd.isna(state["inner_fold"]) else str(state["inner_fold"]).strip()
    return Stage1V2StateSpec(
        state_id=state_id,
        scenario=str(state["scenario"]),
        outer_fold=int(state["outer_fold"]),
        inner_fold=int(inner_text) if inner_text else None,
        state_level=str(state["state_level"]),
        candidate=candidate,
        training_observation_count=counts["training"],
        validation_observation_count=counts["validation"],
        embargo_observation_count=counts["embargo"],
        outer_test_identifier_count=counts["outer_test_identifiers"],
        pedigree=_pedigree_spec(root, state_id),
        marker_factors=tuple(markers),
        projection_environment=projection,
        historical_environment_registry_path=historical_registry,
        h_seeds=h_spec,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight a Stage-1 v2 training state")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--state-id", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = load_state_spec(args.root, args.state_id, args.candidate)
    payload = {
        "status": "PASS",
        "protocol_version": "stage1_v2_trainer_interface_v1",
        "interface_mode": "identifier_and_component_preflight_only",
        **asdict(spec),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
