from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_HELPERS = {
    "canonical_split_mode",
    "grouped_holdout",
    "cv0_split",
    "make_split",
    "group_kfold_splits",
    "split_leakage_record",
}


def definitions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def test_split_utils_is_only_split_logic_implementation() -> None:
    split_definitions = definitions(ROOT / "server_training_pipeline" / "split_utils.py")
    validation_definitions = definitions(ROOT / "server_training_pipeline" / "run_validation_ablation_suite.py")
    trainer_definitions = definitions(ROOT / "server_training_pipeline" / "train_multikernel_gxe_tf.py")

    assert CANONICAL_HELPERS.issubset(split_definitions)
    assert CANONICAL_HELPERS.isdisjoint(validation_definitions)
    assert "split_indices" not in trainer_definitions


def test_validation_suite_imports_shared_make_split() -> None:
    source = (ROOT / "server_training_pipeline" / "run_validation_ablation_suite.py").read_text(encoding="utf-8")
    assert "from .split_utils import canonical_split_mode, group_kfold_splits, make_split, split_leakage_record" in source
