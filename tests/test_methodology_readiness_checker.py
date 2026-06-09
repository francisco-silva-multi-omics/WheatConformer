from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "05_check_model_methodology_readiness.py"
SPEC = importlib.util.spec_from_file_location("methodology_readiness", SCRIPT)
assert SPEC and SPEC.loader
readiness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(readiness)


def touch(root: Path, relative_path: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def create_baseline(root: Path, trait: str = "Grain-Yield") -> None:
    for _, relative_path in readiness.BASELINE_REQUIRED:
        touch(root, relative_path)
    slug = readiness.sanitize_trait_name(trait)
    for filename in readiness.TRAIT_BASELINE_FILES:
        touch(root, f"trained_models/validation_ablation/{slug}/{filename}")


def test_strict_baseline_passes_without_future_components(tmp_path: Path) -> None:
    create_baseline(tmp_path)
    rows = readiness.readiness_rows(tmp_path, ["Grain-Yield"])
    assert readiness.baseline_missing(rows) == []
    graph = next(row for row in rows if row["component"] == "pangenome_graph_gfa")
    assert graph["category"] == "future_thesis_component"
    assert graph["exists"] is False


def test_missing_leakage_summary_fails_baseline_readiness(tmp_path: Path) -> None:
    create_baseline(tmp_path)
    slug = readiness.sanitize_trait_name("Grain-Yield")
    (tmp_path / f"trained_models/validation_ablation/{slug}/split_leakage_summary.tsv").unlink()
    missing = readiness.baseline_missing(readiness.readiness_rows(tmp_path, ["Grain-Yield"]))
    assert any("split_leakage_summary.tsv" in str(row["path"]) for row in missing)
