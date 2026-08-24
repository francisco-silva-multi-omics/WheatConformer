from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import numpy as np
import pandas as pd
from scipy import linalg, sparse, stats
from sklearn.utils.extmath import randomized_svd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

from .stage1_v2_trainer_interface import (
    H_SEEDS,
    PARITY,
    PHASE5,
    PROJECTION,
    load_selection_protocol,
    load_state_spec,
)


PHENOTYPES = Path(
    "audit/v2/phase4_namespace_corrected_release_v1/"
    "corrected_promoted_phenotypes.parquet"
)
INFORMATION_MASKS = PHASE5 / "model_inputs/information_class_masks.parquet"
AUTHORITATIVE_WEIGHTS = PHASE5 / "model_inputs/authoritative_weights.parquet"
KA_NODE_REGISTRY = PHASE5 / "pedigree/pedigree_node_registry.tsv"
PHASE1_ROOT = Path("model_kernels/stage1_v2_phase6_phase1_v2")
EXECUTION_PROTOCOL = Path(
    "server_training_pipeline/stage1_v2_phase6_execution_protocol_v2.json"
)
FACTOR_CACHE_VERSION = "stage1_v2_phase6_factor_cache_v2"
MISSING_DOSAGE = 255
GUARD_REPLAY_ENV = "STAGE1_V2_PHASE1_GUARD_REPLAY"
GUARD_REPLAY_PROTOCOL = "stage1_v2_phase6_phase1_guard_replay_v1"


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*values: object) -> int:
    text = "\x1f".join(str(value) for value in values)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")


def identifier_signature(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype("string").fillna("").str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "pass"}
    )


def guard_replay_enabled() -> bool:
    return os.environ.get(GUARD_REPLAY_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def git_commit(root: Path) -> str:
    code_root = Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=code_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or "Unable to resolve Git commit")
    return process.stdout.strip()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def state_row(root: Path, state_id: str) -> pd.Series:
    registry = pd.read_csv(root / PARITY / "splits/state_registry.tsv", sep="\t", dtype=str)
    selected = registry.loc[registry["state_id"].eq(state_id)]
    if len(selected) != 1:
        raise ValueError(f"Expected one state row for {state_id}; observed={len(selected)}")
    return selected.iloc[0]


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
    env_training = observations["environment_id"].astype(str).isin(training_environments)
    training = outer_training & gid_training & env_training
    if scenario == "GNEW_EOBS":
        validation = outer_training & ~gid_training & env_training
    elif scenario == "GOBS_ENEW":
        validation = outer_training & ~env_training
    elif scenario == "GNEW_ENEW":
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
            values = observations["year"].astype("string").fillna("")
            entity_type = "NORMALIZED_YEAR"
        elif scenario == "COUNTRY_HOLDOUT":
            values = observations["country"].astype("string").fillna("")
            entity_type = "COUNTRY"
        else:
            raise ValueError(f"Unsupported scenario: {scenario}")
        mapping = local.loc[local["entity_type"].eq(entity_type)].set_index("entity_id")[
            "assignment"
        ]
        validation = outer_training & values.map(mapping).fillna("").eq(
            "INNER_VALIDATION_ID_ONLY"
        )
    embargo = ~(training | validation | outer_test)
    if bool((training & validation).any()) or bool((training & outer_test).any()):
        raise ValueError("Stage-1 v2 role masks overlap")
    return training, validation, embargo


def load_state_observations(root: Path, state_id: str) -> tuple[pd.DataFrame, dict[str, object]]:
    state = state_row(root, state_id)
    inner_text = "" if pd.isna(state["inner_fold"]) else str(state["inner_fold"]).strip()
    if not inner_text:
        raise ValueError("Phase-6 model selection requires an inner state")
    scenario = str(state["scenario"])
    outer_fold = int(state["outer_fold"])
    inner_fold = int(inner_text)
    role_column = f"{scenario.lower()}_outer{outer_fold}_role"
    split_path = root / PHASE5 / "splits/observation_split_assignment.parquet"
    columns = [
        "phase4_adjusted_row_id",
        "canonical_gid",
        "environment_id",
        "trial_id",
        "cycle",
        "year",
        "country",
        "trait",
        "primary_weighted_training_eligible",
    ]
    if scenario in {"GNEW_EOBS", "GOBS_ENEW", "GNEW_ENEW"}:
        split = pd.read_parquet(split_path, columns=[*columns, role_column])
    else:
        split = pd.read_parquet(split_path, columns=columns)
        parity_roles = pd.read_parquet(
            root / PARITY / "splits/scenario_observation_roles.parquet",
            columns=["phase4_adjusted_row_id", role_column],
        )
        split = split.merge(
            parity_roles,
            on="phase4_adjusted_row_id",
            how="left",
            validate="one_to_one",
        )
    split = split.loc[bool_series(split["primary_weighted_training_eligible"])].copy()
    training_gids = set(
        pd.read_csv(root / PARITY / str(state["training_gid_path"]), sep="\t", dtype=str)[
            "canonical_gid"
        ].astype(str)
    )
    training_environments = set(
        pd.read_csv(
            root / PARITY / str(state["training_environment_path"]), sep="\t", dtype=str
        )["environment_id"].astype(str)
    )
    assignments = None
    if scenario in {"TEMPORAL_YEAR", "COUNTRY_HOLDOUT"}:
        assignments = pd.read_csv(
            root / PARITY / "splits/inner_entity_assignment.tsv", sep="\t", dtype=str
        )
    train_mask, validation_mask, embargo_mask = state_role_masks(
        split,
        scenario=scenario,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        training_gids=training_gids,
        training_environments=training_environments,
        assignments=assignments,
    )
    selected = split.loc[train_mask | validation_mask].copy()
    selected["selection_role"] = np.where(
        train_mask.loc[selected.index], "TRAINING", "INNER_VALIDATION"
    )
    selected_ids = selected[["phase4_adjusted_row_id"]].drop_duplicates()
    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("selected_ids", selected_ids)
        phenotype = connection.execute(
            """
            SELECT p.phase4_adjusted_row_id, p.adjusted_value
            FROM read_parquet(?) AS p
            INNER JOIN selected_ids AS s USING (phase4_adjusted_row_id)
            """,
            [str((root / PHENOTYPES).resolve())],
        ).fetch_df()
    finally:
        connection.close()
    if phenotype["phase4_adjusted_row_id"].duplicated().any():
        raise ValueError("Selected phenotype identifiers are not unique")
    weights = pd.read_parquet(root / AUTHORITATIVE_WEIGHTS)
    masks = pd.read_parquet(root / INFORMATION_MASKS)
    frame = selected.merge(
        phenotype, on="phase4_adjusted_row_id", how="left", validate="one_to_one"
    ).merge(
        weights[["phase4_stable_observation_id", "authoritative_weight"]],
        left_on="phase4_adjusted_row_id",
        right_on="phase4_stable_observation_id",
        how="left",
        validate="one_to_one",
    ).merge(
        masks[
            [
                "phase4_stable_observation_id",
                "pedigree_available",
                "hibap35k_production_marker_available",
                "haplotype_candidate_available",
                "targeted_marker_available",
                "genotype_information_class",
            ]
        ],
        on="phase4_stable_observation_id",
        how="left",
        validate="one_to_one",
    )
    frame["adjusted_value"] = pd.to_numeric(frame["adjusted_value"], errors="coerce")
    frame["authoritative_weight"] = pd.to_numeric(
        frame["authoritative_weight"], errors="coerce"
    )
    frame = frame.loc[np.isfinite(frame["adjusted_value"])].copy()
    frame["authoritative_weight"] = frame["authoritative_weight"].fillna(0.0).clip(lower=0.0)
    frame = frame.sort_values("phase4_adjusted_row_id").reset_index(drop=True)
    metadata = {
        "state_id": state_id,
        "scenario": scenario,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "training_rows": int(frame["selection_role"].eq("TRAINING").sum()),
        "validation_rows": int(frame["selection_role"].eq("INNER_VALIDATION").sum()),
        "embargo_identifier_rows": int(embargo_mask.sum()),
        "outer_test_identifier_rows": int(
            split[role_column].astype(str).isin({"TEST", "OUTER_TEST_ID_ONLY"}).sum()
        ),
        "training_gid_signature": str(state["training_gid_signature"]),
        "training_environment_signature": str(state["training_environment_signature"]),
        "selected_observation_signature": identifier_signature(
            frame["phase4_adjusted_row_id"].tolist()
        ),
    }
    if metadata["training_rows"] == 0 or metadata["validation_rows"] == 0:
        raise ValueError(f"State lacks training or validation observations: {metadata}")
    return frame, metadata


@dataclass
class FactorBlock:
    name: str
    axis: str
    entity_ids: np.ndarray
    values: np.ndarray
    available: np.ndarray
    eligible_traits: tuple[str, ...] = ()
    state_hash: str = ""

    def validate(self) -> None:
        if self.values.ndim != 2 or len(self.entity_ids) != len(self.values):
            raise ValueError(f"Invalid factor dimensions for {self.name}")
        if self.available.shape != (len(self.values),):
            raise ValueError(f"Invalid availability mask for {self.name}")
        if not np.isfinite(self.values).all():
            raise ValueError(f"Non-finite factor values for {self.name}")
        if len(set(self.entity_ids.astype(str))) != len(self.entity_ids):
            raise ValueError(f"Duplicate factor entities for {self.name}")


def _cache_path(root: Path, state_id: str, label: str, rank: int) -> Path:
    safe = "".join(character if character.isalnum() else "_" for character in label)
    return root / PHASE1_ROOT / "factor_cache" / state_id / f"{safe}__rank{rank}.npz"


def _save_factor(path: Path, factor: FactorBlock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        cache_version=np.asarray([FACTOR_CACHE_VERSION]),
        name=np.asarray([factor.name]),
        axis=np.asarray([factor.axis]),
        entity_ids=factor.entity_ids.astype(str),
        values=factor.values.astype(np.float32),
        available=factor.available.astype(np.bool_),
        eligible_traits=np.asarray(factor.eligible_traits, dtype=str),
        state_hash=np.asarray([factor.state_hash]),
    )
    temporary.replace(path)


def _load_factor(path: Path) -> FactorBlock:
    cached = np.load(path, allow_pickle=False)
    if str(cached["cache_version"][0]) != FACTOR_CACHE_VERSION:
        raise ValueError(f"Stale factor cache: {path}")
    factor = FactorBlock(
        name=str(cached["name"][0]),
        axis=str(cached["axis"][0]),
        entity_ids=cached["entity_ids"].astype(str),
        values=cached["values"].astype(np.float32),
        available=cached["available"].astype(bool),
        eligible_traits=tuple(cached["eligible_traits"].astype(str).tolist()),
        state_hash=str(cached["state_hash"][0]),
    )
    factor.validate()
    return factor


def _cached_factor(
    root: Path,
    state_id: str,
    label: str,
    rank: int,
    builder,
) -> FactorBlock:
    path = _cache_path(root, state_id, label, rank)
    if path.is_file():
        try:
            return _load_factor(path)
        except (KeyError, ValueError):
            pass
    factor = builder()
    factor.validate()
    _save_factor(path, factor)
    return factor


def build_ka_factor(root: Path, state_id: str, rank: int) -> FactorBlock:
    spec = load_state_spec(root, state_id, "ka_identity_location_baseline").pedigree

    def builder() -> FactorBlock:
        entities = pd.read_csv(root / spec.entity_order_path, sep="\t", dtype=str)
        nodes = pd.read_csv(root / KA_NODE_REGISTRY, sep="\t", dtype=str)
        node_map = nodes.set_index("node_id")["node_index"].astype(int)
        node_indices = entities["canonical_gid"].map(node_map)
        if node_indices.isna().any():
            raise ValueError("K_A entity axis contains an unmapped pedigree node")
        operator = sparse.load_npz(root / spec.factor_path).tocsr()
        d_values = np.load(root / spec.mendelian_variance_path, mmap_mode="r")
        weighted = (
            operator[node_indices.astype(int).to_numpy(), :]
            .multiply(np.sqrt(d_values))
            .tocsr()
            / math.sqrt(spec.training_scale)
        )
        training = entities["partition"].eq("TRAINING").to_numpy()
        components = min(rank, int(training.sum()) - 1, weighted.shape[1] - 1)
        if components < 2:
            raise ValueError("K_A has insufficient training support")
        _, _, right = randomized_svd(
            weighted[training, :],
            n_components=components,
            n_iter=3,
            random_state=stable_seed(state_id, "K_A", components),
        )
        values = np.asarray(weighted @ right.T, dtype=np.float32)
        values -= values[training].mean(axis=0, keepdims=True)
        return FactorBlock(
            name="K_A",
            axis="genotype",
            entity_ids=entities["canonical_gid"].astype(str).to_numpy(),
            values=values,
            available=np.ones(len(entities), dtype=bool),
            state_hash=spec.state_hash,
        )

    return _cached_factor(root, state_id, "K_A", rank, builder)


def centered_marker_random_features(
    dosage: np.ndarray,
    *,
    marker_indices: np.ndarray,
    allele_frequency: np.ndarray,
    denominator: float,
    rank: int,
    seed: int,
    marker_major: bool,
    sample_indices: np.ndarray | None = None,
    block_size: int = 1024,
) -> np.ndarray:
    if len(marker_indices) != len(allele_frequency):
        raise ValueError("Marker indices and allele frequencies disagree")
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("Marker denominator must be finite and positive")
    sample_count = dosage.shape[1] if marker_major else dosage.shape[0]
    samples = (
        np.arange(sample_count, dtype=np.int64)
        if sample_indices is None
        else np.asarray(sample_indices, dtype=np.int64)
    )
    output = np.zeros((len(samples), rank), dtype=np.float64)
    rng = np.random.default_rng(seed)
    random_scale = 1.0 / math.sqrt(rank)
    for start in range(0, len(marker_indices), block_size):
        stop = min(len(marker_indices), start + block_size)
        indices = marker_indices[start:stop]
        frequency = allele_frequency[start:stop].astype(np.float64)
        if marker_major:
            block = np.asarray(dosage[np.ix_(indices, samples)], dtype=np.float64).T
        else:
            block = np.asarray(dosage[np.ix_(samples, indices)], dtype=np.float64)
        missing = block == MISSING_DOSAGE
        means = 2.0 * frequency
        block[missing] = np.broadcast_to(means, block.shape)[missing]
        block -= means
        projection = rng.choice([-random_scale, random_scale], size=(len(indices), rank))
        output += block @ projection
    return (output / math.sqrt(denominator)).astype(np.float32)


def build_marker_factor(root: Path, state_id: str, component: str, rank: int) -> FactorBlock:
    candidate = (
        "ka_seeds_historical_environment"
        if component == "K_G_SEEDS_DARTSEQ_V2"
        else "ka_cimmyt_preqc_historical_environment"
    )
    spec = load_state_spec(root, state_id, candidate).marker_factors[0]

    def builder() -> FactorBlock:
        axis = pd.read_csv(root / spec.entity_axis_path, sep="\t", dtype=str)
        dosage = np.load(root / spec.raw_dosage_path, mmap_mode="r")
        parameters = np.load(root / spec.parameter_path, allow_pickle=False)
        marker_indices = parameters["retained_marker_index"].astype(np.int64)
        frequency_key = (
            "allele_frequency"
            if "allele_frequency" in parameters.files
            else "training_allele_frequency"
        )
        denominator_key = (
            "denominator" if "denominator" in parameters.files else "vanraden_denominator"
        )
        allele_frequency = parameters[frequency_key].astype(np.float64)
        denominator = float(parameters[denominator_key].reshape(-1)[0])
        if component == "K_G_SEEDS_DARTSEQ_V2":
            entity_ids = axis["canonical_gid"].astype(str).to_numpy()
            marker_major = False
            sample_indices = None
        else:
            entity_ids = axis["canonical_gid"].astype(str).to_numpy()
            marker_major = True
            sample_indices = axis["shared_call_matrix_column"].astype(int).to_numpy()
        values = centered_marker_random_features(
            dosage,
            marker_indices=marker_indices,
            allele_frequency=allele_frequency,
            denominator=denominator,
            rank=rank,
            seed=stable_seed(state_id, component, rank),
            marker_major=marker_major,
            sample_indices=sample_indices,
        )
        state = state_row(root, state_id)
        training_gids = set(
            pd.read_csv(root / PARITY / str(state["training_gid_path"]), sep="\t", dtype=str)[
                "canonical_gid"
            ].astype(str)
        )
        training = np.asarray([gid in training_gids for gid in entity_ids], dtype=bool)
        if not training.any():
            raise ValueError(f"No training marker entities for {component}")
        values -= values[training].mean(axis=0, keepdims=True)
        return FactorBlock(
            name=component,
            axis="genotype",
            entity_ids=entity_ids,
            values=values,
            available=np.ones(len(entity_ids), dtype=bool),
            state_hash=spec.state_hash,
        )

    return _cached_factor(root, state_id, component, rank, builder)


def build_h_seeds_factor(root: Path, state_id: str, rank: int) -> FactorBlock:
    h_spec = load_state_spec(root, state_id, "h_seeds_historical_environment").h_seeds
    if h_spec is None:
        raise ValueError("H_SEEDS specification is absent")
    if not h_spec.component_available:
        factor = build_ka_factor(root, state_id, rank)
        factor.name = "H_SEEDS_MASKED_TO_K_A"
        factor.state_hash = h_spec.state_hash
        return factor

    def builder() -> FactorBlock:
        pedigree = load_state_spec(root, state_id, "ka_identity_location_baseline").pedigree
        entities = pd.read_csv(root / pedigree.entity_order_path, sep="\t", dtype=str)
        nodes = pd.read_csv(root / KA_NODE_REGISTRY, sep="\t", dtype=str)
        node_map = nodes.set_index("node_id")["node_index"].astype(int)
        node_indices = entities["canonical_gid"].map(node_map).astype(int).to_numpy()
        operator = sparse.load_npz(root / pedigree.factor_path).tocsr()
        d_values = np.load(root / pedigree.mendelian_variance_path, mmap_mode="r")
        weighted_all = operator[node_indices, :].multiply(np.sqrt(d_values)).tocsr()
        arrays = np.load(root / h_spec.state_path, allow_pickle=False)
        overlap_nodes = arrays["pedigree_node_indices"].astype(np.int64)
        overlap_seed_rows = arrays["seeds_row_indices"].astype(np.int64)
        weighted_overlap = (
            operator[overlap_nodes, :].multiply(np.sqrt(d_values)).tocsr()
        )
        a22 = np.asarray((weighted_overlap @ weighted_overlap.T).toarray(), dtype=np.float64)
        a22 = (a22 + a22.T) * 0.5
        chol = linalg.cho_factor(a22, lower=True, check_finite=False)

        def propagate(values: np.ndarray) -> np.ndarray:
            solved = linalg.cho_solve(chol, values, check_finite=False)
            coefficients = weighted_overlap.T @ solved
            return np.asarray(weighted_all @ coefficients, dtype=np.float64)

        rng = np.random.default_rng(stable_seed(state_id, "H_SEEDS_PEDIGREE", rank))
        projection = rng.choice(
            [-1.0 / math.sqrt(rank), 1.0 / math.sqrt(rank)],
            size=(weighted_all.shape[1], rank),
        )
        all_pedigree = np.asarray(weighted_all @ projection, dtype=np.float64)
        overlap_pedigree = np.asarray(weighted_overlap @ projection, dtype=np.float64)
        propagated_pedigree = propagate(overlap_pedigree)
        ka_scale = float(arrays["pedigree_training_scale"].reshape(-1)[0])
        conditional = (all_pedigree - propagated_pedigree) / math.sqrt(ka_scale)
        pedigree_blend = math.sqrt(1.0 - h_spec.genomic_blend_weight) * (
            propagated_pedigree / math.sqrt(ka_scale)
        )
        seeds = build_marker_factor(root, state_id, "K_G_SEEDS_DARTSEQ_V2", rank)
        overlap_genomic = seeds.values[overlap_seed_rows]
        genomic_scale = float(arrays["genomic_scale"].reshape(-1)[0])
        genomic = math.sqrt(h_spec.genomic_blend_weight * genomic_scale) * propagate(
            overlap_genomic
        )
        combined = np.concatenate([conditional, pedigree_blend, genomic], axis=1)
        training = entities["partition"].eq("TRAINING").to_numpy()
        combined -= combined[training].mean(axis=0, keepdims=True)
        components = min(rank, int(training.sum()) - 1, combined.shape[1] - 1)
        _, _, right = randomized_svd(
            combined[training],
            n_components=components,
            n_iter=3,
            random_state=stable_seed(state_id, "H_SEEDS_FINAL", components),
        )
        values = np.asarray(combined @ right.T, dtype=np.float32)
        return FactorBlock(
            name="H_SEEDS",
            axis="genotype",
            entity_ids=entities["canonical_gid"].astype(str).to_numpy(),
            values=values,
            available=np.ones(len(entities), dtype=bool),
            state_hash=h_spec.state_hash,
        )

    return _cached_factor(root, state_id, "H_SEEDS", rank, builder)


def _environment_axis(root: Path, state_id: str) -> pd.DataFrame:
    registry = pd.read_csv(root / PHASE5 / "environment/ke_registry.tsv", sep="\t", dtype=str)
    row = registry.loc[
        registry["state_id"].eq(state_id) & registry["component"].eq("K_E_identity")
    ]
    if len(row) != 1:
        raise ValueError(f"Missing K_E identity axis for {state_id}")
    return pd.read_csv(root / PHASE5 / str(row.iloc[0]["entity_order_path"]), sep="\t")


def build_identity_geo_factors(root: Path, state_id: str, rank: int) -> tuple[FactorBlock, ...]:
    def identity_builder() -> FactorBlock:
        axis = _environment_axis(root, state_id)
        training = axis["partition"].eq("TRAINING").to_numpy()
        values = np.zeros((len(axis), rank), dtype=np.float32)
        rng = np.random.default_rng(stable_seed(state_id, "K_E_IDENTITY", rank))
        values[training] = rng.choice(
            [-1.0 / math.sqrt(rank), 1.0 / math.sqrt(rank)],
            size=(int(training.sum()), rank),
        )
        values[training] -= values[training].mean(axis=0, keepdims=True)
        return FactorBlock(
            name="K_E_IDENTITY",
            axis="environment",
            entity_ids=axis["environment_id"].astype(str).to_numpy(),
            values=values,
            available=training,
            state_hash=identifier_signature(axis.loc[training, "environment_id"]),
        )

    def geo_builder() -> FactorBlock:
        axis = _environment_axis(root, state_id)
        training = axis["partition"].eq("TRAINING").to_numpy()
        training_levels = set(axis.loc[training, "location_key"].astype(str))
        available = axis["location_key"].astype(str).isin(training_levels).to_numpy()
        levels = sorted(training_levels)
        rng = np.random.default_rng(stable_seed(state_id, "K_E_EXACT_LOCATION", rank))
        level_values = rng.choice(
            [-1.0 / math.sqrt(rank), 1.0 / math.sqrt(rank)], size=(len(levels), rank)
        ).astype(np.float32)
        lookup = {value: index for index, value in enumerate(levels)}
        values = np.zeros((len(axis), rank), dtype=np.float32)
        positions = axis["location_key"].astype(str).map(lookup)
        valid = positions.notna().to_numpy()
        values[valid] = level_values[positions.loc[valid].astype(int)]
        values -= values[training].mean(axis=0, keepdims=True)
        values[~available] = 0.0
        return FactorBlock(
            name="K_E_EXACT_LOCATION",
            axis="environment",
            entity_ids=axis["environment_id"].astype(str).to_numpy(),
            values=values,
            available=available,
            state_hash=identifier_signature(levels),
        )

    identity = _cached_factor(root, state_id, "K_E_IDENTITY", rank, identity_builder)
    geo = _cached_factor(root, state_id, "K_E_EXACT_LOCATION", rank, geo_builder)
    return identity, geo


WINDOW_LABELS = {
    "ESTABLISHMENT_D0_30": "d0_30",
    "VEGETATIVE_D30_60": "d30_60",
    "REPRODUCTIVE_D60_90": "d60_90",
    "GRAIN_FILL_EARLY_D90_120": "d90_120",
    "GRAIN_FILL_LATE_D120_150": "d120_150",
    "LATE_SEASON_D150_180": "d150_180",
}


def _component_window(component: str) -> str | None:
    for label, window in WINDOW_LABELS.items():
        if label in component:
            return window
    return None


def _source_values(
    root: Path,
    parameters: pd.DataFrame,
    environment_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    source_path = root / str(parameters.iloc[0]["source_path"])
    if source_path.suffix.lower() == ".parquet":
        source = pd.read_parquet(source_path)
    else:
        source = pd.read_csv(source_path, sep="\t", low_memory=False)
    component = str(parameters.iloc[0]["component"])
    window = _component_window(component)
    values = np.zeros((len(environment_ids), len(parameters)), dtype=np.float64)
    source_available = np.zeros(len(environment_ids), dtype=bool)
    environment_lookup = {value: index for index, value in enumerate(environment_ids)}
    for column_index, row in enumerate(parameters.itertuples(index=False)):
        feature = str(row.feature)
        local_window = window
        source_feature = feature
        if "__" in feature and feature.split("__", 1)[0] in set(WINDOW_LABELS.values()):
            local_window, source_feature = feature.split("__", 1)
        local = source
        if local_window is not None:
            local = source.loc[source["window_label"].astype(str).eq(local_window)]
        if source_feature not in local.columns:
            raise ValueError(f"Source feature is absent: {component}/{feature}")
        indexed = local[["env_id", source_feature]].copy()
        indexed[source_feature] = pd.to_numeric(indexed[source_feature], errors="coerce")
        conflicts = indexed.groupby("env_id", sort=False)[source_feature].nunique(
            dropna=True
        )
        if bool(conflicts.gt(1).any()):
            raise ValueError(
                f"Conflicting duplicate source values: {component}/{feature}; "
                f"environments={int(conflicts.gt(1).sum())}"
            )
        series = indexed.groupby("env_id", sort=False)[source_feature].first()
        mapped = pd.Series(environment_ids).map(series)
        observed = mapped.notna().to_numpy()
        source_available |= observed
        raw = mapped.to_numpy(dtype=np.float64)
        raw[~np.isfinite(raw)] = float(row.imputation_median)
        sd = float(row.scaling_sd_after_imputation)
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Invalid scaling SD: {component}/{feature}")
        values[:, column_index] = (
            (raw - float(row.centering_mean_after_imputation)) / sd
        ) * float(row.factor_postscale)
    values[~source_available] = 0.0
    return values, source_available


def build_historical_environment(
    root: Path, state_id: str, rank: int
) -> tuple[tuple[FactorBlock, ...], np.ndarray, np.ndarray, np.ndarray]:
    cache = _cache_path(root, state_id, "HISTORICAL_ENVIRONMENT", rank)
    if cache.is_file():
        saved = np.load(cache, allow_pickle=False)
        if str(saved["cache_version"][0]) == FACTOR_CACHE_VERSION:
            axis = saved["entity_ids"].astype(str)
            factors = saved["factor_values"].astype(np.float32)
            availability = saved["factor_available"].astype(bool)
            starts = saved["factor_starts"].astype(int)
            stops = saved["factor_stops"].astype(int)
            names = saved["component_names"].astype(str)
            hashes = saved["component_hashes"].astype(str)
            blocks = tuple(
                FactorBlock(
                    name,
                    "environment",
                    axis,
                    factors[:, start:stop],
                    availability[:, index],
                    eligible_traits=("1000_GRAIN_WEIGHT",)
                    if name == "K_E_TGW_FIXED_GRAIN_FILL"
                    else (),
                    state_hash=hashes[index],
                )
                for index, (name, start, stop) in enumerate(zip(names, starts, stops))
            )
            for block in blocks:
                block.validate()
            result = (
                blocks,
                saved["reaction_design"].astype(np.float32),
                saved["reaction_available"].astype(bool),
                axis,
            )
            saved.close()
            return result
        saved.close()
    axis_frame = _environment_axis(root, state_id)
    environment_ids = axis_frame["environment_id"].astype(str).to_numpy(dtype=str)
    training = axis_frame["partition"].eq("TRAINING").to_numpy()
    parameters = pd.read_csv(
        root / PARITY / "environment/environment_preprocessing_parameters.tsv",
        sep="\t",
    )
    parameters = parameters.loc[parameters["state_id"].eq(state_id)].copy()
    blocks: list[FactorBlock] = []
    stage_values: list[np.ndarray] = []
    stage_available: list[np.ndarray] = []
    for component, group in parameters.groupby("component", sort=True):
        values, available = _source_values(root, group, environment_ids)
        training_available = training & available
        components = min(rank, values.shape[1], int(training_available.sum()) - 1)
        if components < 1:
            raise ValueError(
                f"Historical component lacks training support: {state_id}/{component}"
            )
        if values.shape[1] == 1:
            factor = values.astype(np.float32)
        else:
            _, _, right = randomized_svd(
                values[training_available],
                n_components=components,
                n_iter=3,
                random_state=stable_seed(state_id, component, components),
            )
            factor = np.asarray(values @ right.T, dtype=np.float32)
        factor -= factor[training_available].mean(axis=0, keepdims=True)
        factor[~available] = 0.0
        component_hash = hashlib.sha256(
            group.to_csv(index=False, lineterminator="\n").encode("utf-8")
        ).hexdigest()
        blocks.append(
            FactorBlock(
                component,
                "environment",
                environment_ids,
                factor,
                available,
                eligible_traits=("1000_GRAIN_WEIGHT",)
                if component == "K_E_TGW_FIXED_GRAIN_FILL"
                else (),
                state_hash=component_hash,
            )
        )
        if component.startswith("K_E_STAGE_"):
            stage_values.append(values)
            stage_available.append(available)
    if not blocks or not stage_values:
        raise ValueError(f"Historical parity components are incomplete: {state_id}")
    reaction_design = np.concatenate(stage_values, axis=1).astype(np.float32)
    reaction_available = np.logical_and.reduce(stage_available)
    reaction_design[~reaction_available] = 0.0
    factor_starts = np.cumsum([0, *[block.values.shape[1] for block in blocks[:-1]]])
    factor_stops = factor_starts + np.asarray(
        [block.values.shape[1] for block in blocks]
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        cache_version=np.asarray([FACTOR_CACHE_VERSION]),
        entity_ids=environment_ids,
        component_names=np.asarray([block.name for block in blocks], dtype=str),
        component_hashes=np.asarray([block.state_hash for block in blocks], dtype=str),
        factor_starts=factor_starts,
        factor_stops=factor_stops,
        factor_values=np.concatenate([block.values for block in blocks], axis=1),
        factor_available=np.column_stack([block.available for block in blocks]),
        reaction_design=reaction_design,
        reaction_available=reaction_available,
    )
    temporary.replace(cache)
    return tuple(blocks), reaction_design, reaction_available, environment_ids


def build_projection_environment(
    root: Path, state_id: str
) -> tuple[tuple[FactorBlock, ...], np.ndarray, np.ndarray, np.ndarray]:
    spec = load_state_spec(root, state_id, "ka_projection_core").projection_environment
    if spec is None:
        raise ValueError("Projection environment specification is absent")
    entities = pd.read_csv(root / spec.entity_axis_path, sep="\t", dtype=str)
    factor = np.load(root / spec.kernel_factor_path, mmap_mode="r").astype(np.float32)
    features = np.load(root / spec.standardized_features_path, mmap_mode="r").astype(
        np.float32
    )
    active = bool_series(entities["component_active"]).to_numpy()
    factor[~active] = 0.0
    features[~active] = 0.0
    block = FactorBlock(
        "E_PROJECTION_CORE_V1",
        "environment",
        entities["environment_id"].astype(str).to_numpy(),
        factor,
        active,
        state_hash=sha256_file(root / spec.kernel_factor_path),
    )
    block.validate()
    return (block,), features, active, block.entity_ids


def build_candidate_factors(
    root: Path,
    state_id: str,
    candidate: str,
    configuration: dict[str, object],
) -> tuple[tuple[FactorBlock, ...], tuple[FactorBlock, ...], np.ndarray, np.ndarray, np.ndarray]:
    genotype_rank = int(configuration["max_rank_genotype"])
    environment_rank = int(configuration["max_rank_environment"])
    if candidate.startswith("h_seeds_"):
        genotype = [build_h_seeds_factor(root, state_id, genotype_rank)]
    else:
        genotype = [build_ka_factor(root, state_id, genotype_rank)]
        if "seeds" in candidate:
            genotype.append(
                build_marker_factor(root, state_id, "K_G_SEEDS_DARTSEQ_V2", genotype_rank)
            )
        if "cimmyt_preqc" in candidate:
            genotype.append(
                build_marker_factor(root, state_id, "K_G_CIMMYT_PRE_QC", genotype_rank)
            )
    if candidate == "ka_identity_location_baseline":
        environment = build_identity_geo_factors(root, state_id, environment_rank)
        reaction_design = np.concatenate([block.values for block in environment], axis=1)
        reaction_available = np.logical_or.reduce([block.available for block in environment])
        environment_ids = environment[0].entity_ids
    elif candidate.endswith("projection_core"):
        environment, reaction_design, reaction_available, environment_ids = (
            build_projection_environment(root, state_id)
        )
    else:
        base = build_identity_geo_factors(root, state_id, environment_rank)
        historical, reaction_design, reaction_available, environment_ids = (
            build_historical_environment(root, state_id, environment_rank)
        )
        environment = (*base, *historical)
    return tuple(genotype), tuple(environment), reaction_design, reaction_available, environment_ids


def _candidate_marker_gids(
    root: Path,
    state_id: str,
    candidate: str,
    protocol: dict[str, object],
) -> set[str]:
    candidate_protocol = protocol["candidates"][candidate]
    components = set(candidate_protocol["genotype_components"])
    marker_gids: set[str] = set()
    if components.intersection({"K_G_SEEDS_DARTSEQ_V2", "H_SEEDS"}):
        seeds_spec = load_state_spec(
            root, state_id, "ka_seeds_historical_environment"
        ).marker_factors[0]
        seeds_axis = pd.read_csv(root / seeds_spec.entity_axis_path, sep="\t", dtype=str)
        marker_gids.update(seeds_axis["canonical_gid"].astype(str))
    if "K_G_CIMMYT_PRE_QC" in components:
        cimmyt_spec = load_state_spec(
            root, state_id, "ka_cimmyt_preqc_historical_environment"
        ).marker_factors[0]
        cimmyt_axis = pd.read_csv(root / cimmyt_spec.entity_axis_path, sep="\t", dtype=str)
        marker_gids.update(cimmyt_axis["canonical_gid"].astype(str))
    return marker_gids


def _projection_active_environments(root: Path, state_id: str) -> set[str]:
    spec = load_state_spec(root, state_id, "ka_projection_core").projection_environment
    if spec is None:
        raise ValueError("Projection environment specification is absent")
    axis = pd.read_csv(root / spec.entity_axis_path, sep="\t", dtype=str)
    active = bool_series(axis["component_active"])
    return set(axis.loc[active, "environment_id"].astype(str))


def reporting_subset_masks(
    frame: pd.DataFrame,
    *,
    marker_gids: set[str],
    projection_active_environments: set[str],
) -> dict[str, pd.Series]:
    marker_supported = frame["canonical_gid"].astype(str).isin(marker_gids)
    pedigree_supported = bool_series(frame["pedigree_available"])
    projection_active = frame["environment_id"].astype(str).isin(
        projection_active_environments
    )
    return {
        "PEDIGREE_ONLY": pedigree_supported & ~marker_supported,
        "MARKER_SUPPORTED": marker_supported,
        "PEDIGREE_AND_MARKER": pedigree_supported & marker_supported,
        "NEITHER_PRODUCTION_PEDIGREE_NOR_DENSE_MARKERS": (
            ~pedigree_supported & ~marker_supported
        ),
        "RECOVERED_IDENTITY_OR_COMPONENT": marker_supported,
        "PROJECTION_CORE_ACTIVE": projection_active,
        "PROJECTION_CORE_INACTIVE_814_ENVIRONMENTS": ~projection_active,
    }


def build_reporting_masks(
    root: Path,
    state_id: str,
    frame: pd.DataFrame,
    protocol: dict[str, object],
) -> dict[str, dict[str, pd.Series]]:
    projection_active = _projection_active_environments(root, state_id)
    candidates = protocol["candidate_stages"]["phase_1_individual"]
    return {
        candidate: reporting_subset_masks(
            frame,
            marker_gids=_candidate_marker_gids(root, state_id, candidate, protocol),
            projection_active_environments=projection_active,
        )
        for candidate in candidates
    }


def add_factor_indices(
    frame: pd.DataFrame,
    genotype: Sequence[FactorBlock],
    environment: Sequence[FactorBlock],
    reaction_environment_ids: Sequence[str],
    reaction_available: np.ndarray,
) -> tuple[list[str], list[str]]:
    genotype_columns: list[str] = []
    environment_columns: list[str] = []
    for block_index, block in enumerate(genotype):
        column = f"genotype_factor_{block_index}_index"
        lookup = {value: index for index, value in enumerate(block.entity_ids.astype(str))}
        frame[column] = frame["canonical_gid"].astype(str).map(lookup).fillna(-1).astype(np.int32)
        available = frame[column].ge(0)
        if available.any():
            indices = frame.loc[available, column].to_numpy(dtype=np.int64)
            local_available = block.available[indices]
            frame.loc[available, column] = np.where(local_available, indices, -1).astype(
                np.int32
            )
        genotype_columns.append(column)
    for block_index, block in enumerate(environment):
        column = f"environment_factor_{block_index}_index"
        lookup = {value: index for index, value in enumerate(block.entity_ids.astype(str))}
        frame[column] = frame["environment_id"].astype(str).map(lookup).fillna(-1).astype(np.int32)
        available = frame[column].ge(0)
        if available.any():
            indices = frame.loc[available, column].to_numpy(dtype=np.int64)
            local_available = block.available[indices]
            frame.loc[available, column] = np.where(local_available, indices, -1).astype(
                np.int32
            )
        environment_columns.append(column)
    reaction_lookup = {
        value: index for index, value in enumerate(np.asarray(reaction_environment_ids).astype(str))
    }
    frame["reaction_environment_index"] = (
        frame["environment_id"].astype(str).map(reaction_lookup).fillna(-1).astype(np.int32)
    )
    available = frame["reaction_environment_index"].ge(0)
    if available.any():
        indices = frame.loc[available, "reaction_environment_index"].to_numpy(dtype=np.int64)
        frame.loc[available, "reaction_environment_index"] = np.where(
            reaction_available[indices], indices, -1
        ).astype(np.int32)
    return genotype_columns, environment_columns


class Stage1V2ReactionNorm(tf.keras.Model):
    def __init__(
        self,
        *,
        genotype: Sequence[FactorBlock],
        environment: Sequence[FactorBlock],
        reaction_design: np.ndarray,
        trait_names: Sequence[str],
        latent_dim: int,
        reaction_rank: int,
        residual_floor: float,
        weight_decay: float,
        seed: int,
        reaction_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.genotype_blocks = tuple(genotype)
        self.environment_blocks = tuple(environment)
        self.trait_names = tuple(trait_names)
        self.latent_dim = latent_dim
        self.reaction_rank = reaction_rank
        self.residual_floor = residual_floor
        self.weight_decay = weight_decay
        self.reaction_enabled = reaction_enabled
        self.genotype_factors = [tf.constant(block.values) for block in genotype]
        self.environment_factors = [tf.constant(block.values) for block in environment]
        self.reaction_design = tf.constant(reaction_design, dtype=tf.float32)
        def initializer(label: str) -> tf.keras.initializers.Initializer:
            return tf.keras.initializers.RandomNormal(
                stddev=0.02, seed=stable_seed(seed, label)
            )

        self.intercept = self.add_weight(
            name="trait_intercept", shape=(len(trait_names),), initializer="zeros"
        )
        self.raw_residual = self.add_weight(
            name="raw_residual_scale",
            shape=(len(trait_names),),
            initializer=tf.keras.initializers.Constant(0.5),
        )
        self.trait_loadings = self.add_weight(
            name="trait_loadings",
            shape=(latent_dim, len(trait_names)),
            initializer=initializer("trait_loadings"),
        )
        self.genotype_main = [
            self.add_weight(
                name=f"g_main_{index}",
                shape=(block.values.shape[1], latent_dim),
                initializer=initializer(f"g_main_{index}_{block.name}"),
            )
            for index, block in enumerate(genotype)
        ]
        self.environment_main = [
            self.add_weight(
                name=f"e_main_{index}",
                shape=(block.values.shape[1], latent_dim),
                initializer=initializer(f"e_main_{index}_{block.name}"),
            )
            for index, block in enumerate(environment)
        ]
        self.environment_eligibility = []
        trait_lookup = {value: index for index, value in enumerate(trait_names)}
        for block in environment:
            eligibility = np.ones(len(trait_names), dtype=bool)
            if block.eligible_traits:
                eligibility[:] = False
                for trait in block.eligible_traits:
                    eligibility[trait_lookup[trait]] = True
            self.environment_eligibility.append(tf.constant(eligibility))
        self.genotype_reaction_projection = []
        self.genotype_reaction_coefficients = []
        if reaction_enabled:
            for index, block in enumerate(genotype):
                rng = np.random.default_rng(stable_seed(seed, block.name, "reaction"))
                projection = rng.choice(
                    [
                        -1.0 / math.sqrt(block.values.shape[1]),
                        1.0 / math.sqrt(block.values.shape[1]),
                    ],
                    size=(block.values.shape[1], reaction_rank),
                ).astype(np.float32)
                self.genotype_reaction_projection.append(
                    tf.constant(block.values @ projection)
                )
                self.genotype_reaction_coefficients.append(
                    self.add_weight(
                        name=f"g_reaction_{index}",
                        shape=(reaction_rank, latent_dim),
                        initializer=initializer(f"g_reaction_{index}_{block.name}"),
                    )
                )
            self.environment_reaction_coefficients = self.add_weight(
                name="environment_reaction",
                shape=(reaction_design.shape[1], latent_dim),
                initializer=initializer("environment_reaction"),
            )
        else:
            self.environment_reaction_coefficients = None

    def residual_scales(self) -> tf.Tensor:
        return tf.nn.softplus(self.raw_residual) + self.residual_floor

    def call(self, inputs, training: bool = False):
        genotype_indices, environment_indices, reaction_index, trait_index = inputs
        trait_loading = tf.gather(tf.transpose(self.trait_loadings), trait_index)
        prediction = tf.gather(self.intercept, trait_index)
        for index, (factor, coefficients) in enumerate(
            zip(self.genotype_factors, self.genotype_main)
        ):
            local_index = genotype_indices[:, index]
            available = local_index >= 0
            gathered = tf.gather(factor, tf.maximum(local_index, 0))
            latent = tf.matmul(gathered, coefficients)
            effect = tf.reduce_sum(latent * trait_loading, axis=1)
            prediction += effect * tf.cast(available, tf.float32)
        for index, (factor, coefficients) in enumerate(
            zip(self.environment_factors, self.environment_main)
        ):
            local_index = environment_indices[:, index]
            available = local_index >= 0
            gathered = tf.gather(factor, tf.maximum(local_index, 0))
            latent = tf.matmul(gathered, coefficients)
            effect = tf.reduce_sum(latent * trait_loading, axis=1)
            eligible = tf.gather(self.environment_eligibility[index], trait_index)
            prediction += effect * tf.cast(available & eligible, tf.float32)
        if self.reaction_enabled:
            reaction_available = reaction_index >= 0
            environment_design = tf.gather(
                self.reaction_design, tf.maximum(reaction_index, 0)
            )
            environment_latent = tf.matmul(
                environment_design, self.environment_reaction_coefficients
            )
            for index, (features, coefficients) in enumerate(
                zip(
                    self.genotype_reaction_projection,
                    self.genotype_reaction_coefficients,
                )
            ):
                local_index = genotype_indices[:, index]
                available = (local_index >= 0) & reaction_available
                gathered = tf.gather(features, tf.maximum(local_index, 0))
                genotype_latent = tf.matmul(gathered, coefficients)
                effect = tf.reduce_sum(
                    genotype_latent * environment_latent * trait_loading, axis=1
                ) / math.sqrt(self.latent_dim)
                prediction += effect * tf.cast(available, tf.float32)
        return prediction

    def regularization_loss(self) -> tf.Tensor:
        terms = [
            tf.reduce_sum(tf.square(value))
            for value in self.trainable_variables
            if "raw_residual" not in value.name and "trait_intercept" not in value.name
        ]
        return self.weight_decay * tf.add_n(terms)


def make_dataset(
    frame: pd.DataFrame,
    genotype_columns: Sequence[str],
    environment_columns: Sequence[str],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    inputs = (
        frame[list(genotype_columns)].to_numpy(dtype=np.int32),
        frame[list(environment_columns)].to_numpy(dtype=np.int32),
        frame["reaction_environment_index"].to_numpy(dtype=np.int32),
        frame["trait_index"].to_numpy(dtype=np.int32),
    )
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            inputs,
            frame["y_scaled"].to_numpy(dtype=np.float32),
            frame["loss_weight"].to_numpy(dtype=np.float32),
        )
    )
    if shuffle:
        dataset = dataset.shuffle(
            min(len(frame), 100_000), seed=seed, reshuffle_each_iteration=True
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def predict(
    model: Stage1V2ReactionNorm,
    frame: pd.DataFrame,
    genotype_columns: Sequence[str],
    environment_columns: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    dataset = make_dataset(
        frame,
        genotype_columns,
        environment_columns,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    )
    values = [model(inputs, training=False) for inputs, _, _ in dataset]
    return (
        tf.concat(values, axis=0).numpy()
        if values
        else np.empty(0, dtype=np.float32)
    )


def macro_nrmse(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    scores = []
    for trait_index, positions in frame.groupby("trait_index", sort=True).groups.items():
        index = np.asarray(list(positions), dtype=np.int64)
        error = prediction[index] - frame.loc[index, "y_scaled"].to_numpy(dtype=float)
        scores.append(float(np.sqrt(np.mean(np.square(error)))))
    return float(np.mean(scores)) if scores else float("inf")


def prepare_targets(
    frame: pd.DataFrame, trait_names: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trait_lookup = {value: index for index, value in enumerate(trait_names)}
    frame = frame.loc[frame["trait"].isin(trait_lookup)].copy()
    frame["trait_index"] = frame["trait"].map(trait_lookup).astype(np.int32)
    training = frame["selection_role"].eq("TRAINING")
    rows = []
    frame["y_scaled"] = np.nan
    frame["loss_weight"] = 0.0
    for trait in trait_names:
        local_training = training & frame["trait"].eq(trait)
        values = frame.loc[local_training, "adjusted_value"].to_numpy(dtype=np.float64)
        weights = frame.loc[local_training, "authoritative_weight"].to_numpy(dtype=np.float64)
        positive = weights > 0
        mean = float(np.average(values[positive], weights=weights[positive]))
        variance = float(np.average(np.square(values[positive] - mean), weights=weights[positive]))
        sd = math.sqrt(max(variance, 1e-12))
        local = frame["trait"].eq(trait)
        frame.loc[local, "y_scaled"] = (frame.loc[local, "adjusted_value"] - mean) / sd
        mean_positive_weight = float(np.mean(weights[positive]))
        normalized_weight = frame.loc[local, "authoritative_weight"].to_numpy(dtype=float)
        normalized_weight = np.where(
            normalized_weight > 0,
            normalized_weight / max(mean_positive_weight, 1e-12),
            0.0,
        )
        frame.loc[local, "loss_weight"] = normalized_weight
        rows.append(
            {
                "trait_name_canonical": trait,
                "training_rows": int(local_training.sum()),
                "training_positive_weight_rows": int(positive.sum()),
                "training_weighted_mean": mean,
                "training_weighted_sd": sd,
            }
        )
    return frame, pd.DataFrame(rows)


def validation_reporting_metrics(
    local: pd.DataFrame,
    scaling: pd.DataFrame,
    reporting_masks: dict[str, dict[str, pd.Series]],
) -> pd.DataFrame:
    scale_lookup = scaling.set_index("trait_name_canonical")
    rows: list[dict[str, object]] = []
    for mask_candidate, subset_masks in reporting_masks.items():
        for subset, mask in subset_masks.items():
            local_mask = pd.Series(mask, index=local.index).fillna(False).astype(bool)
            group = local.loc[local_mask]
            if group.empty:
                rows.append(
                    {
                        "mask_candidate": mask_candidate,
                        "subset": subset,
                        "rows": 0,
                        "unique_genotypes": 0,
                        "unique_environments": 0,
                        "trait_count": 0,
                        "observation_id_signature": identifier_signature([]),
                        "normalized_rmse_macro": np.nan,
                        "pearson_macro": np.nan,
                    }
                )
                continue
            errors = []
            correlations = []
            for trait, trait_group in group.groupby("trait", sort=True):
                y = trait_group["adjusted_value"].to_numpy(dtype=float)
                p = trait_group["prediction"].to_numpy(dtype=float)
                sd = float(scale_lookup.loc[trait, "training_weighted_sd"])
                errors.append(float(np.sqrt(np.mean(np.square(p - y))) / sd))
                if len(trait_group) >= 2 and np.std(y) > 0 and np.std(p) > 0:
                    correlations.append(float(np.corrcoef(y, p)[0, 1]))
            identifiers = sorted(group["phase4_adjusted_row_id"].astype(str))
            rows.append(
                {
                    "mask_candidate": mask_candidate,
                    "subset": subset,
                    "rows": len(group),
                    "unique_genotypes": int(group["canonical_gid"].nunique()),
                    "unique_environments": int(group["environment_id"].nunique()),
                    "trait_count": int(group["trait"].nunique()),
                    "observation_id_signature": identifier_signature(identifiers),
                    "normalized_rmse_macro": float(np.mean(errors)),
                    "pearson_macro": (
                        float(np.mean(correlations)) if correlations else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def validation_metrics(
    frame: pd.DataFrame,
    prediction_scaled: np.ndarray,
    scaling: pd.DataFrame,
    reporting_masks: dict[str, dict[str, pd.Series]],
    candidate: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    local = frame.reset_index(drop=True).copy()
    scale_lookup = scaling.set_index("trait_name_canonical")
    local["prediction_scaled"] = prediction_scaled
    local["prediction"] = [
        value * float(scale_lookup.loc[trait, "training_weighted_sd"])
        + float(scale_lookup.loc[trait, "training_weighted_mean"])
        for value, trait in zip(prediction_scaled, local["trait"])
    ]
    trait_rows = []
    for trait, group in local.groupby("trait", sort=True):
        y = group["adjusted_value"].to_numpy(dtype=float)
        p = group["prediction"].to_numpy(dtype=float)
        sd = float(scale_lookup.loc[trait, "training_weighted_sd"])
        rmse = float(np.sqrt(np.mean(np.square(p - y))))
        pearson = float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else np.nan
        slope = float(np.cov(y, p, ddof=0)[0, 1] / np.var(p)) if np.var(p) > 0 else np.nan
        trait_rows.append(
            {
                "trait_name_canonical": trait,
                "rows": len(group),
                "normalized_rmse": rmse / sd,
                "pearson": pearson,
                "calibration_slope": slope,
                "calibration_error": abs(slope - 1.0) if np.isfinite(slope) else np.nan,
            }
        )
    trait_metrics = pd.DataFrame(trait_rows)
    reporting_metrics = validation_reporting_metrics(local, scaling, reporting_masks)
    subset_metrics = reporting_metrics.loc[
        reporting_metrics["mask_candidate"].eq(candidate)
    ].drop(columns="mask_candidate")
    centered_y = local["adjusted_value"] - local.groupby(["trait", "environment_id"])[
        "adjusted_value"
    ].transform("mean")
    centered_p = local["prediction"] - local.groupby(["trait", "environment_id"])[
        "prediction"
    ].transform("mean")
    rank_y = centered_y.groupby([local["trait"], local["environment_id"]]).rank()
    rank_p = centered_p.groupby([local["trait"], local["environment_id"]]).rank()
    valid_rank = rank_y.notna() & rank_p.notna()
    centered_spearman = (
        float(np.corrcoef(rank_y[valid_rank], rank_p[valid_rank])[0, 1])
        if valid_rank.sum() >= 2
        else np.nan
    )
    rng = np.random.default_rng(20260822)
    correct = 0
    pairs = 0
    for _, group in local.groupby(["trait", "environment_id"], sort=False):
        if len(group) < 2:
            continue
        count = min(100, len(group) * (len(group) - 1) // 2)
        left = rng.integers(0, len(group), size=count)
        right = rng.integers(0, len(group), size=count)
        keep = left != right
        if not keep.any():
            continue
        y = group["adjusted_value"].to_numpy(dtype=float)
        p = group["prediction"].to_numpy(dtype=float)
        observed = np.sign(y[left[keep]] - y[right[keep]])
        predicted = np.sign(p[left[keep]] - p[right[keep]])
        non_ties = observed != 0
        correct += int((observed[non_ties] == predicted[non_ties]).sum())
        pairs += int(non_ties.sum())
    summary = {
        "validation_macro_normalized_rmse": float(trait_metrics["normalized_rmse"].mean()),
        "validation_macro_pearson": float(trait_metrics["pearson"].mean()),
        "validation_macro_calibration_error": float(
            trait_metrics["calibration_error"].mean()
        ),
        "within_environment_centered_spearman": centered_spearman,
        "within_environment_pairwise_accuracy": correct / pairs if pairs else np.nan,
        "within_environment_pair_count": pairs,
    }
    return trait_metrics, subset_metrics, reporting_metrics, summary


def train_run(
    *,
    root: Path,
    state_id: str,
    candidate: str,
    configuration_label: str,
    seed: int,
    out_dir: Path,
) -> dict[str, object]:
    root = root.resolve()
    out_dir = out_dir.resolve()
    intra_op_threads = int(os.environ.get("STAGE1_V2_INTRA_OP_THREADS", "16"))
    inter_op_threads = int(os.environ.get("STAGE1_V2_INTER_OP_THREADS", "2"))
    if intra_op_threads < 1 or inter_op_threads < 1:
        raise ValueError("TensorFlow thread counts must be positive")
    tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
    tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
    protocol = load_selection_protocol(root)
    configuration = protocol["hyperparameter_configurations"][configuration_label]
    spec = load_state_spec(root, state_id, candidate)
    if spec.state_level != "INNER":
        raise ValueError("Phase-1 training requires an inner state")
    frame, role_metadata = load_state_observations(root, state_id)
    trait_names = [*protocol["primary_traits"], *protocol["exploratory_traits"]]
    frame, scaling = prepare_targets(frame, trait_names)
    genotype, environment, reaction_design, reaction_available, reaction_ids = (
        build_candidate_factors(root, state_id, candidate, configuration)
    )
    genotype_columns, environment_columns = add_factor_indices(
        frame,
        genotype,
        environment,
        reaction_ids,
        reaction_available,
    )
    full_reporting_masks = build_reporting_masks(root, state_id, frame, protocol)
    current_masks = full_reporting_masks[candidate]
    frame["candidate_marker_supported"] = current_masks["MARKER_SUPPORTED"]
    frame["recovered_component_supported"] = current_masks[
        "RECOVERED_IDENTITY_OR_COMPONENT"
    ]
    frame["projection_core_active"] = current_masks["PROJECTION_CORE_ACTIVE"]
    training = frame.loc[frame["selection_role"].eq("TRAINING")].reset_index(drop=True)
    validation = frame.loc[
        frame["selection_role"].eq("INNER_VALIDATION")
    ].reset_index(drop=True)
    validation_reporting_masks = build_reporting_masks(
        root, state_id, validation, protocol
    )
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    model = Stage1V2ReactionNorm(
        genotype=genotype,
        environment=environment,
        reaction_design=reaction_design,
        trait_names=trait_names,
        latent_dim=int(configuration["latent_dim"]),
        reaction_rank=int(configuration["reaction_rank"]),
        residual_floor=0.05,
        weight_decay=float(configuration["weight_decay"]),
        seed=seed,
    )
    optimizer = tf.keras.optimizers.Adam(float(configuration["learning_rate"]))
    batch_size = int(configuration["batch_size"])
    train_dataset = make_dataset(
        training,
        genotype_columns,
        environment_columns,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    @tf.function
    def train_step(inputs, target, weight):
        with tf.GradientTape() as tape:
            prediction = model(inputs, training=True)
            trait_index = inputs[3]
            scale = tf.gather(model.residual_scales(), trait_index)
            nll = 0.5 * tf.square((target - prediction) / scale) + tf.math.log(scale)
            denominator = tf.maximum(tf.reduce_sum(weight), 1e-6)
            loss = tf.reduce_sum(nll * weight) / denominator + model.regularization_loss()
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    best_metric = float("inf")
    best_weights: list[np.ndarray] | None = None
    epochs_without_improvement = 0
    epoch_rows = []
    epochs_max = int(configuration["epochs_max"])
    patience = int(configuration["early_stopping_patience"])
    for epoch in range(1, epochs_max + 1):
        losses = []
        for inputs, target, weight in train_dataset:
            losses.append(train_step(inputs, target, weight))
        mean_training_loss = float(tf.reduce_mean(tf.stack(losses)).numpy())
        if epoch == 1 or epoch % 5 == 0:
            prediction_scaled = predict(
                model,
                validation,
                genotype_columns,
                environment_columns,
                batch_size,
            )
            metric = macro_nrmse(validation, prediction_scaled)
            row = {
                "epoch": epoch,
                "train_gaussian_nll_regularized": mean_training_loss,
                "validation_macro_normalized_rmse": metric,
            }
            epoch_rows.append(row)
            print(json.dumps(row), flush=True)
            if metric < best_metric - 1e-7:
                best_metric = metric
                best_weights = model.get_weights()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 5 if epoch > 1 else 1
            if epochs_without_improvement >= patience:
                break
    if best_weights is None:
        raise RuntimeError("Training did not produce a finite validation checkpoint")
    model.set_weights(best_weights)
    prediction_scaled = predict(
        model, validation, genotype_columns, environment_columns, batch_size
    )
    trait_metrics, subset_metrics, reporting_metrics, summary = validation_metrics(
        validation,
        prediction_scaled,
        scaling,
        validation_reporting_masks,
        candidate,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    scaling.to_csv(out_dir / "trait_scaling.tsv", sep="\t", index=False)
    pd.DataFrame(epoch_rows).to_csv(out_dir / "epoch_history.tsv", sep="\t", index=False)
    trait_metrics.to_csv(out_dir / "validation_trait_metrics.tsv", sep="\t", index=False)
    subset_metrics.to_csv(out_dir / "validation_subset_metrics.tsv", sep="\t", index=False)
    reporting_metrics.to_csv(
        out_dir / "validation_guard_metrics.tsv", sep="\t", index=False
    )
    factor_rows = [
        {
            "component": block.name,
            "axis": block.axis,
            "entities": len(block.entity_ids),
            "available_entities": int(block.available.sum()),
            "rank": block.values.shape[1],
            "state_hash": block.state_hash,
        }
        for block in (*genotype, *environment)
    ]
    pd.DataFrame(factor_rows).to_csv(
        out_dir / "active_component_factors.tsv", sep="\t", index=False
    )
    metadata = {
        "status": "PASS",
        "protocol_version": (
            GUARD_REPLAY_PROTOCOL
            if guard_replay_enabled()
            else "stage1_v2_phase6_tf_trainer_v3_matched_guard_masks"
        ),
        "stage1_version": "Stage-1 v2",
        "state_id": state_id,
        "scenario": spec.scenario,
        "outer_fold": spec.outer_fold,
        "inner_fold": spec.inner_fold,
        "candidate": candidate,
        "configuration_label": configuration_label,
        "configuration": configuration,
        "seed": seed,
        "execution_backend": os.environ.get("STAGE1_V2_EXECUTION_BACKEND", "wsl_gpu"),
        "intra_op_threads": intra_op_threads,
        "inter_op_threads": inter_op_threads,
        **role_metadata,
        **summary,
        "best_validation_macro_nrmse": best_metric,
        "epochs_completed": int(epoch_rows[-1]["epoch"]),
        "selection_protocol_sha256": sha256_file(
            Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
            / "server_training_pipeline/stage1_v2_phase6_selection_protocol_v1.json"
        ),
        "execution_protocol_sha256": sha256_file(
            Path(os.environ.get("WHEATCONFORMER_CODE_ROOT", root)).resolve()
            / EXECUTION_PROTOCOL
        ),
        "trainer_sha256": sha256_file(Path(__file__)),
        "code_commit": git_commit(root),
        "validation_observation_signature": identifier_signature(
            validation["phase4_adjusted_row_id"].tolist()
        ),
        "guard_mask_candidate_count": int(reporting_metrics["mask_candidate"].nunique()),
        "guard_mask_observation_signatures_written": True,
        "h_seeds_direct_marker_support_included": True,
        "projection_core_mask_candidate_independent": True,
        "phenotype_values_read": True,
        "inner_validation_metrics_read": True,
        "outer_test_outcomes_read": False,
        "outer_test_metrics_read": False,
        "final_holdout_outcomes_read": False,
        "model_training_performed": True,
    }
    write_json(out_dir / "run_metadata.json", metadata)
    tf.keras.backend.clear_session()
    del model
    gc.collect()
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one frozen Stage-1 v2 Phase-6 inner run")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--state-id", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_run(
        root=args.root,
        state_id=args.state_id,
        candidate=args.candidate,
        configuration_label=args.configuration,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
