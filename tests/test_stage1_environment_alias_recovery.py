from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from audit.build_stage1_environment_alias_registry import build_alias_registry
from build_stage1_model_kernels import apply_environment_aliases


ROOT = Path(__file__).resolve().parents[1]


def recovery_table(source_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "env_kernel_id": source_ids,
            "recovery_readiness": ["P1_RECOVER_ENVIRONMENT"] * len(source_ids),
            "stage1_rows": [10] * len(source_ids),
            "unique_genotypes": [2] * len(source_ids),
            "unique_traits": [3] * len(source_ids),
            "represented_raw_plot_records": [12] * len(source_ids),
        }
    )


def test_trial_alias_resolves_ambiguous_nontrial_identity() -> None:
    source = [
        "SHORT|1|101|MEXICO|SITE A|2020",
        "SHORT|2|102|MEXICO|SITE B|2020",
        "SHORT|3|103|MEXICO|SITE C|2020",
        "SHORT|4|104|MEXICO|SITE D|2020",
    ]
    order = pd.DataFrame(
        {
            "env_id": [
                "EXPANDED TRIAL|1|101|MEXICO|SITE A|2020",
                "EXPANDED TRIAL|2|102|MEXICO|SITE B|2020",
                "EXPANDED TRIAL|3|103|MEXICO|SITE C|2020",
                "EXPANDED TRIAL|4|104|MEXICO|SITE D|2020",
                "OTHER TRIAL|4|104|MEXICO|SITE D|2020",
            ]
        }
    )

    registry, review, aliases = build_alias_registry(
        recovery_table(source), order, minimum_anchors=3, minimum_share=0.95
    )

    assert len(registry) == 4
    assert set(registry["mapping_status"]) == {"ACCEPTED_ALIAS"}
    assert registry["target_env_id"].str.startswith("EXPANDED TRIAL|").all()
    assert (
        registry["match_class"] == "trial_alias_resolved_nontrial_collision"
    ).sum() == 1
    assert review["source_env_id"].is_unique
    assert aliases.loc[0, "trial_alias_status"] == "ACCEPTED_DOMINANT_TRIAL_ALIAS"
    assert aliases.loc[0, "unique_identity_anchor_count"] == 3


def test_trial_alias_does_not_resolve_without_minimum_anchor_support() -> None:
    source = ["SHORT|1|101|MEXICO|SITE A|2020"]
    order = pd.DataFrame(
        {"env_id": ["EXPANDED TRIAL|1|101|MEXICO|SITE A|2020"]}
    )

    registry, review, aliases = build_alias_registry(
        recovery_table(source), order, minimum_anchors=3, minimum_share=0.95
    )

    assert registry.empty
    assert review.loc[0, "mapping_status"] == "NO_ACCEPTED_TRIAL_ALIAS"
    assert aliases.loc[0, "trial_alias_status"] == "REQUIRES_TRIAL_ALIAS_REVIEW"


def test_apply_environment_aliases_preserves_original_ids() -> None:
    frame = pd.DataFrame({"env_kernel_id": ["short", "existing"]})
    aliases = pd.DataFrame(
        {
            "source_env_id": ["short"],
            "target_env_id": ["expanded"],
            "mapping_status": ["ACCEPTED_ALIAS"],
        }
    )

    result, stats = apply_environment_aliases(frame, "env_kernel_id", aliases)

    assert result["env_kernel_id"].tolist() == ["expanded", "existing"]
    assert result["env_kernel_id_original"].tolist() == ["short", "existing"]
    assert result["environment_alias_applied"].tolist() == [True, False]
    assert result["environment_alias_mapping_status"].tolist() == [
        "ACCEPTED_ALIAS",
        "NOT_APPLICABLE",
    ]
    assert stats == {
        "environment_alias_registry_rows": 1,
        "environment_alias_applied_rows": 1,
        "environment_alias_applied_source_ids": 1,
    }


def test_stage1_builder_applies_certified_environment_alias(tmp_path: Path) -> None:
    phenotype_path = tmp_path / "phenotypes.tsv"
    pd.DataFrame(
        {
            "panel_sample_id": ["g1", "g1"],
            "env_kernel_id": ["short", "existing"],
            "y_tilde_g_e": [1.0, 2.0],
            "SE_g_e": [0.1, 0.1],
            "var_g_e": [0.01, 0.01],
            "weight_g_e": [1.0, 1.0],
            "trait_name_canonical": ["Trait", "Trait"],
            "stage1_model_status": [
                "linear_model_adjusted",
                "linear_model_adjusted",
            ],
        }
    ).to_csv(phenotype_path, sep="\t", index=False)
    genotype_order = tmp_path / "genotype_order.tsv"
    environment_order = tmp_path / "environment_order.tsv"
    pd.DataFrame({"sample_id": ["g1"]}).to_csv(
        genotype_order, sep="\t", index=False
    )
    pd.DataFrame({"env_id": ["existing", "expanded"]}).to_csv(
        environment_order, sep="\t", index=False
    )
    genotype_kernel = tmp_path / "K_G.npy"
    environment_kernel = tmp_path / "K_E.npy"
    np.save(genotype_kernel, np.ones((1, 1), dtype=np.float32))
    np.save(environment_kernel, np.eye(2, dtype=np.float32))
    alias_path = tmp_path / "aliases.tsv"
    pd.DataFrame(
        {
            "source_env_id": ["short"],
            "target_env_id": ["expanded"],
            "mapping_status": ["ACCEPTED_ALIAS"],
        }
    ).to_csv(alias_path, sep="\t", index=False)
    out = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "build_stage1_model_kernels.py"),
            "--stage1-phenotypes",
            str(phenotype_path),
            "--geno-kernel",
            str(genotype_kernel),
            "--geno-order",
            str(genotype_order),
            "--env-kernel",
            str(environment_kernel),
            "--env-order",
            str(environment_order),
            "--environment-alias-map",
            str(alias_path),
            "--out-dir",
            str(out),
            "--prefix",
            "toy",
            "--write-tsv",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    observations = pd.read_csv(
        out / "toy_model_ready_stage1_observations.tsv.gz", sep="\t"
    )
    summary = pd.read_csv(out / "toy_model_kernel_summary.tsv", sep="\t")
    metrics = dict(zip(summary["metric"], summary["value"], strict=True))

    assert observations["env_kernel_id"].tolist() == ["expanded", "existing"]
    assert observations["env_kernel_id_original"].tolist() == ["short", "existing"]
    assert observations["env_kernel_index"].tolist() == [1, 0]
    assert int(metrics["environment_alias_applied_rows"]) == 1


def test_alias_model_validator_accepts_exact_partition(tmp_path: Path) -> None:
    readiness = tmp_path / "readiness.tsv"
    pd.DataFrame(
        {
            "canonical_observation_id": ["retained", "recovered", "weight_bad"],
            "env_kernel_id": ["existing", "short", "short"],
            "recovery_readiness": [
                "RETAINED_REFERENCE",
                "P1_RECOVER_ENVIRONMENT",
                "P3_REPAIR_WEIGHT_METADATA",
            ],
        }
    ).to_csv(readiness, sep="\t", index=False)
    model = tmp_path / "model.tsv"
    pd.DataFrame(
        {
            "canonical_observation_id": ["retained", "recovered"],
            "canonical_germplasm_key": ["g1", "g1"],
            "env_kernel_id": ["existing", "expanded"],
            "env_kernel_id_original": ["existing", "short"],
            "environment_alias_applied": [False, True],
            "environment_alias_mapping_status": [
                "NOT_APPLICABLE",
                "ACCEPTED_ALIAS",
            ],
            "geno_kernel_index": [0, 0],
            "env_kernel_index": [0, 1],
        }
    ).to_csv(model, sep="\t", index=False)
    aliases = tmp_path / "aliases.tsv"
    pd.DataFrame(
        {
            "source_env_id": ["short"],
            "target_env_id": ["expanded"],
            "mapping_status": ["ACCEPTED_ALIAS"],
        }
    ).to_csv(aliases, sep="\t", index=False)
    genotype_order = tmp_path / "genotype_order.tsv"
    environment_order = tmp_path / "environment_order.tsv"
    pd.DataFrame({"sample_id": ["g1"]}).to_csv(
        genotype_order, sep="\t", index=False
    )
    pd.DataFrame({"env_id": ["existing", "expanded"]}).to_csv(
        environment_order, sep="\t", index=False
    )
    out = tmp_path / "validation"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "audit.validate_stage1_environment_alias_recovery",
            "--root",
            str(tmp_path),
            "--readiness-ledger",
            str(readiness),
            "--alias-registry",
            str(aliases),
            "--model-observations",
            str(model),
            "--genotype-order",
            str(genotype_order),
            "--environment-order",
            str(environment_order),
            "--out-dir",
            str(out),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    validation = pd.read_csv(
        out / "stage1_environment_alias_model_validation.tsv", sep="\t"
    )
    assert set(validation["status"]) == {"PASS"}
