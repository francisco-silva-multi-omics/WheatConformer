from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


RELEASE_ID = "P5PESP_20260809_V2_274E41DF"
RELEASE_RELATIVE = Path("audit/v2/phase5_panel_environment_scenario_parity_extension_v2")
UPSTREAM_RELEASE_ID = "P5SBK_20260808_V1_274E41DF"
UPSTREAM_RELATIVE = Path("audit/v2/phase5_split_bound_kernel_validation_v2")
V1_INCIDENT_RELATIVE = Path("audit/v2/phase5_panel_environment_scenario_parity_extension_v1")
BUNDLE_RELATIVE = Path("server_phase5_parity_bundle")
SEED = 20260809


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def index_signature(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_tsv(path: Path, value: pd.DataFrame | Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(list(value))
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


@dataclass(frozen=True)
class AccessDecision:
    relative_path: str
    decision: str
    matched_rule: str
    operation: str


class ProtectedPathGuard:
    """Fail-closed path guard for every bundle access and protected repository path."""

    def __init__(self, repository_root: Path, denylist_path: Path):
        self.repository_root = repository_root.resolve()
        self.bundle_root = (self.repository_root / BUNDLE_RELATIVE).resolve()
        raw_rules = denylist_path.read_text(encoding="utf-8").splitlines()
        self.rules = tuple(line.strip().replace("\\", "/") for line in raw_rules if line.strip())
        self.decisions: list[AccessDecision] = []

    def _relative(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.repository_root).as_posix()
        except ValueError:
            return resolved.as_posix()

    def matched_rule(self, path: Path) -> str:
        relative = self._relative(path)
        lowered = relative.lower()
        for rule in self.rules:
            normalized = rule.lower()
            if fnmatch.fnmatchcase(lowered, normalized):
                return rule
            if normalized.startswith("**/") and fnmatch.fnmatchcase(lowered, normalized[3:]):
                return rule
        return ""

    def assert_allowed(self, path: Path, operation: str = "READ") -> Path:
        resolved = path.resolve()
        rule = self.matched_rule(resolved)
        relative = self._relative(resolved)
        if rule:
            self.decisions.append(AccessDecision(relative, "DENY", rule, operation))
            raise PermissionError(f"Protected-path denylist blocks {operation}: {relative} ({rule})")
        self.decisions.append(AccessDecision(relative, "ALLOW", "", operation))
        return resolved

    def inventory_metadata_only(self, path: Path) -> AccessDecision:
        relative = self._relative(path)
        rule = self.matched_rule(path)
        decision = AccessDecision(relative, "METADATA_ONLY", rule, "INVENTORY_FILENAME_SIZE_SHA256_METADATA")
        self.decisions.append(decision)
        return decision

    def audit_frame(self) -> pd.DataFrame:
        rows = [decision.__dict__ for decision in self.decisions]
        if not rows:
            return pd.DataFrame(columns=["relative_path", "decision", "matched_rule", "operation"])
        return pd.DataFrame(rows).drop_duplicates().sort_values(["relative_path", "operation"])


def normalize_cycle_year(value: object) -> int:
    text = str(value).strip()
    if len(text) == 4 and text.isdigit():
        return int(text)
    if len(text) == 5 and text[2] == "-" and text[:2].isdigit() and text[3:].isdigit():
        end = int(text[3:])
        return 1900 + end if int(text[:2]) >= 70 else 2000 + end
    raise ValueError(f"Unsupported cycle/year label: {value!r}")


def deterministic_balanced_assignment(
    entity_weights: Mapping[str, int], folds: int, namespace: str
) -> dict[str, int]:
    totals = {fold: 0 for fold in range(1, folds + 1)}
    counts = {fold: 0 for fold in range(1, folds + 1)}
    ordered = sorted(
        entity_weights,
        key=lambda entity: (
            -int(entity_weights[entity]),
            hashlib.sha256(f"{SEED}|{namespace}|{entity}".encode()).hexdigest(),
        ),
    )
    assignment: dict[str, int] = {}
    for entity in ordered:
        fold = min(
            totals,
            key=lambda candidate: (
                totals[candidate],
                counts[candidate],
                hashlib.sha256(f"{SEED}|{namespace}|{entity}|{candidate}".encode()).hexdigest(),
            ),
        )
        assignment[entity] = fold
        totals[fold] += int(entity_weights[entity])
        counts[fold] += 1
    return assignment


def contiguous_weighted_blocks(year_weights: Mapping[int, int], blocks: int) -> dict[int, int]:
    years = sorted(year_weights)
    if len(years) < blocks:
        raise ValueError(f"Need at least {blocks} years, found {len(years)}")
    total = sum(int(year_weights[year]) for year in years)
    assignment: dict[int, int] = {}
    cumulative = 0
    for position, year in enumerate(years):
        remaining_years = len(years) - position
        remaining_blocks = blocks - (max(assignment.values()) if assignment else 0)
        target = min(blocks, 1 + int((cumulative * blocks) // max(total, 1)))
        if remaining_years <= remaining_blocks:
            target = blocks - remaining_years + 1
        assignment[year] = max(1, target)
        cumulative += int(year_weights[year])
    # Enforce non-empty, monotone blocks deterministically if weight jumps skipped a block.
    observed = sorted(set(assignment.values()))
    if observed != list(range(1, blocks + 1)):
        boundaries = np.linspace(0, len(years), blocks + 1, dtype=int)
        for block in range(1, blocks + 1):
            for year in years[boundaries[block - 1] : boundaries[block]]:
                assignment[year] = block
    return assignment


def factor_diagnostics(factor: np.ndarray, training_rows: np.ndarray | None = None) -> dict[str, Any]:
    if training_rows is not None:
        factor = factor[training_rows]
    if factor.ndim != 2:
        raise ValueError("Factor must be two-dimensional")
    gram = np.asarray(factor.T @ factor, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh((gram + gram.T) / 2.0)
    positive = eigenvalues[eigenvalues > 1e-12]
    effective_rank = 0.0
    if positive.size:
        probabilities = positive / positive.sum()
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    diagonal = np.einsum("ij,ij->i", factor, factor)
    return {
        "entities": int(factor.shape[0]),
        "factor_columns": int(factor.shape[1]),
        "algebraic_rank": int(np.linalg.matrix_rank(gram, tol=1e-10)),
        "effective_rank": effective_rank,
        "minimum_nonzero_eigenvalue": float(positive.min()) if positive.size else 0.0,
        "maximum_eigenvalue": float(positive.max()) if positive.size else 0.0,
        "mean_diagonal": float(diagonal.mean()) if diagonal.size else 0.0,
        "minimum_diagonal": float(diagonal.min()) if diagonal.size else 0.0,
        "maximum_diagonal": float(diagonal.max()) if diagonal.size else 0.0,
        "all_finite": bool(np.isfinite(factor).all()),
        "psd_by_factor_construction": True,
        "symmetry_by_factor_construction": True,
    }


def kernel_alignment(left: np.ndarray, right: np.ndarray) -> float:
    cross = left.T @ right
    numerator = float(np.sum(cross * cross))
    left_norm = float(np.sum((left.T @ left) ** 2))
    right_norm = float(np.sum((right.T @ right) ** 2))
    denominator = math.sqrt(left_norm * right_norm)
    return numerator / denominator if denominator > 0 else math.nan


def git_head(root: Path) -> str:
    head = root / ".git/HEAD"
    if not head.exists():
        return "UNKNOWN"
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref: "):
        ref = root / ".git" / text[5:]
        if ref.exists():
            return ref.read_text(encoding="utf-8").strip()
    return text


def environment_versions() -> list[str]:
    try:
        from importlib.metadata import distributions

        return sorted(
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in distributions()
            if dist.metadata.get("Name")
        )
    except Exception:
        return []


def ensure_fail_if_exists(path: Path) -> None:
    try:
        os.mkdir(path)
    except FileExistsError as exc:
        raise SystemExit(f"FAIL_IF_EXISTS: release root already exists: {path}") from exc
