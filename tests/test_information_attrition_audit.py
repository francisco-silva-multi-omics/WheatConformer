from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def row(
    observation_id: str,
    gid: str,
    env: str,
    *,
    value: float = 1.0,
    has_environment: bool = True,
    original: str = "DTH",
    trait: str = "DAYS_TO_HEADING",
) -> dict[str, object]:
    return {
        "canonical_observation_id": observation_id,
        "canonical_germplasm_key": gid,
        "germplasm_id": gid,
        "resolved_gid": gid.removeprefix("GID"),
        "env_kernel_id": env,
        "trial_name": "TRIAL",
        "cycle": "2020",
        "occ": "1",
        "loc_no": "10",
        "country": "MEXICO",
        "loc_desc": "SITE",
        "trait_name_canonical": trait,
        "trait_name_original": original,
        "unit": "DAYS",
        "phenotype_value": value,
        "raw_numeric_records": 2,
        "raw_plot_records": 2,
        "n_records": 2,
        "source_level": "raw_plot_linked_summary",
        "gid_resolution_status": "resolved" if gid else "unresolved",
        "genotype_name": gid,
        "has_environment_kernel": has_environment,
    }


def stage_row(observation_id: str, gid: str, env: str, original: str) -> dict[str, object]:
    return {
        "canonical_observation_id": observation_id,
        "canonical_germplasm_key": gid,
        "resolved_gid": gid.removeprefix("GID"),
        "env_kernel_id": env,
        "trait_name_canonical": "DAYS_TO_HEADING",
        "trait_name_original": original,
        "unit": "DAYS",
        "trial_name": "TRIAL",
        "cycle": "2020",
        "country": "MEXICO",
        "y_tilde_g_e": 1.0,
        "stage1_model_status": "linear_model_adjusted",
        "n_plot_records": 2,
    }


def model_row(observation_id: str, gid: str, env: str, original: str) -> dict[str, object]:
    output = stage_row(observation_id, gid, env, original)
    output.pop("y_tilde_g_e")
    output.pop("stage1_model_status")
    output.pop("n_plot_records")
    output.update(
        {
            "phenotype_value": 1.0,
            "geno_kernel_index": 0,
            "env_kernel_index": 0,
        }
    )
    return output


def test_information_attrition_cli_classifies_recovery_opportunities(
    tmp_path: Path,
) -> None:
    canonical = pd.DataFrame(
        [
            row("C1", "GID1", "E1", original="DTH_RETAINED"),
            row("C2", "GID1", "E1", value=np.nan, original="DTH_NONFINITE"),
            row("C3", "", "E1", original="DTH_UNRESOLVED"),
            row("C4", "GID2", "E2", has_environment=False, original="DTH_NO_ENV"),
            row("C5", "GID3", "E1", original="DTH_NO_PEDIGREE"),
            row("C6", "GID1", "E1", original="DTH_NO_STAGE1"),
            row("C7", "GID2", "E1", original="DTH_NO_MODEL"),
            row(
                "C8",
                "GID1",
                "E1",
                original="GY",
                trait="GRAIN_YIELD",
            ),
        ]
    )
    stage1 = pd.DataFrame(
        [
            stage_row("S1", "GID1", "E1", "DTH_RETAINED"),
            stage_row("S7", "GID2", "E1", "DTH_NO_MODEL"),
        ]
    )
    model = pd.DataFrame([model_row("S1", "GID1", "E1", "DTH_RETAINED")])
    ledger = model.copy()
    ledger["genotype_id"] = "GID1"
    ledger["environment_id"] = "E1"

    (tmp_path / "integrated_database").mkdir()
    (tmp_path / "phenotypes").mkdir()
    stage_model = tmp_path / "model_kernels/stage1_pedigree_env"
    final_ledger = tmp_path / "model_kernels/final"
    pedigree = tmp_path / "genotype_panels/pedigree_canonical_v3"
    stage_model.mkdir(parents=True)
    final_ledger.mkdir(parents=True)
    pedigree.mkdir(parents=True)

    canonical.to_parquet(
        tmp_path / "integrated_database/canonical.parquet", index=False
    )
    stage1.to_parquet(tmp_path / "phenotypes/stage1.parquet", index=False)
    model.to_parquet(stage_model / "observations.parquet", index=False)
    ledger.to_parquet(final_ledger / "ledger.parquet", index=False)
    pd.DataFrame(
        {
            "sample_id": ["GID1"],
            "source_kernel_index": [0],
            "compact_kernel_index": [0],
        }
    ).to_csv(stage_model / "g_order.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "env_id": ["E1"],
            "source_kernel_index": [0],
            "compact_kernel_index": [0],
        }
    ).to_csv(stage_model / "e_order.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "sample_id": ["GID1", "GID2"],
            "compact_kernel_index": [0, 1],
        }
    ).to_csv(pedigree / "order.tsv", sep="\t", index=False)

    out_dir = tmp_path / "audit_out"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "audit.audit_information_attrition",
            "--root",
            str(tmp_path),
            "--canonical",
            "integrated_database/canonical.parquet",
            "--stage1-adjusted",
            "phenotypes/stage1.parquet",
            "--stage1-model-observations",
            "model_kernels/stage1_pedigree_env/observations.parquet",
            "--ledger",
            "model_kernels/final/ledger.parquet",
            "--stage1-genotype-order",
            "model_kernels/stage1_pedigree_env/g_order.tsv",
            "--stage1-environment-order",
            "model_kernels/stage1_pedigree_env/e_order.tsv",
            "--canonical-pedigree-order",
            "genotype_panels/pedigree_canonical_v3/order.tsv",
            "--regulatory-manifest",
            "absent.tsv.gz",
            "--out-dir",
            str(out_dir),
            "--trait",
            "DAYS_TO_HEADING",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    loss = pd.read_csv(
        out_dir / "selected_trait_exclusive_loss_summary.tsv", sep="\t"
    ).set_index("exclusive_loss_reason")
    expected = {
        "retained_in_final_ledger_key",
        "nonfinite_target",
        "unresolved_genotype_identity",
        "environment_kernel_unavailable",
        "absent_from_canonical_pedigree",
        "not_reconstructed_by_stage1_raw_pipeline",
        "outside_stage1_genotype_environment_intersection",
    }
    assert expected == set(loss.index)
    assert loss["canonical_rows"].eq(1).all()

    provenance = json.loads(
        (out_dir / "information_attrition_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert provenance["status"] == "PASS"
    assert provenance["outer_test_metrics_read"] is False
    assert provenance["final_holdout_outcomes_read"] is False
    assert provenance["target_values_used_for_selection_or_imputation"] is False
    assert Path(provenance["code_root"]).resolve() == ROOT
    assert provenance["git_commit"] == subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()

    policy = pd.read_csv(out_dir / "imputation_policy.tsv", sep="\t")
    target = policy[policy["information_type"].eq("target_phenotype")].iloc[0]
    assert target["policy"] == "PROHIBITED"
