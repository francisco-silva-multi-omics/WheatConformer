from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_PROTOCOL = Path(__file__).with_name("final_evaluation_protocol.json")
REQUIRED_FAMILIES = {
    "unseen_environments",
    "unseen_genotypes",
    "unseen_genotypes_and_environments",
    "temporal_country_holdout",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path | None = None) -> dict[str, Any]:
    protocol_path = (path or DEFAULT_PROTOCOL).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen":
        raise ValueError(f"Final-evaluation protocol is not frozen: {protocol_path}")
    families = set(protocol.get("generalization_families", {}))
    if families != REQUIRED_FAMILIES:
        raise ValueError(
            "Final-evaluation protocol has the wrong generalization families: "
            f"expected={sorted(REQUIRED_FAMILIES)} observed={sorted(families)}"
        )
    traits = [str(value).strip().upper() for value in protocol.get("traits", [])]
    climatology = {
        str(value).strip().upper()
        for value in protocol.get("climatology_eligible_traits", [])
    }
    if not traits or not climatology or not climatology.issubset(set(traits)):
        raise ValueError("Frozen traits and climatology eligibility are inconsistent")
    if int(protocol.get("outer_folds", 0)) < 2 or int(protocol.get("inner_folds", 0)) < 2:
        raise ValueError("Nested evaluation requires at least two outer and inner folds")
    support = protocol.get("final_holdout_support", {})
    required_support = {
        "minimum_environment_fraction",
        "minimum_environment_count",
        "maximum_environment_fraction",
        "minimum_rows_per_trait",
        "minimum_environments_per_trait",
    }
    if set(support) != required_support:
        raise ValueError(
            "Frozen final-holdout support policy is incomplete: "
            f"expected={sorted(required_support)} observed={sorted(support)}"
        )
    minimum_fraction = float(support["minimum_environment_fraction"])
    maximum_fraction = float(support["maximum_environment_fraction"])
    if not 0 < minimum_fraction <= maximum_fraction < 1:
        raise ValueError("Final-holdout environment fractions are invalid")
    if int(support["minimum_environment_count"]) < 1:
        raise ValueError("Final holdout requires at least one environment")
    if int(support["minimum_rows_per_trait"]) < 1:
        raise ValueError("Final holdout requires positive per-trait row support")
    if int(support["minimum_environments_per_trait"]) < 2:
        raise ValueError("Final holdout requires at least two environments per trait")
    protocol["traits"] = traits
    protocol["climatology_eligible_traits"] = sorted(climatology)
    protocol["protocol_path"] = str(protocol_path)
    protocol["protocol_sha256"] = file_sha256(protocol_path)
    return protocol


def require_non_discovery_seed(seed: int, protocol: dict[str, Any]) -> None:
    forbidden = {int(value) for value in protocol["discovery_seeds_forbidden"]}
    if int(seed) in forbidden:
        raise ValueError(
            f"Seed {seed} was used during discovery and is forbidden by the frozen "
            "final-evaluation protocol"
        )


def climatology_traits_csv(protocol: dict[str, Any]) -> str:
    return ",".join(protocol["climatology_eligible_traits"])
