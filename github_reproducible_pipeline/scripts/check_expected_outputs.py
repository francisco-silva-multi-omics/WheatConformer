from __future__ import annotations

from pathlib import Path


EXPECTED = [
    "metadata_outputs/all_trials_genotype_manifest_resolved.tsv",
    "metadata_outputs/usable_trial_to_canonical_hmp_matches.tsv",
    "genotype_panels/hmp/hmp_sample_by_marker.QCfiltered.parquet",
    "genotype_panels/hmp/K_HMP.QCfiltered.npy",
    "genotype_panels/hmp/hmp_K_sample_order.QCfiltered.tsv",
    "phenotypes/stage1_adjusted_phenotypes.parquet",
    "environment/K_E.npy",
    "environment/env_kernel_sample_order.tsv",
    "integrated_database/canonical_trial_genotype_environment_plot_table.parquet",
    "model_kernels/stage1_hmp_env/stage1_hmp_env_model_ready_stage1_observations.parquet",
]

OPTIONAL = [
    "genotype_panels/diversity_80k/diversity_80k_marker_prior_features.parquet",
    "genotype_panels/dartseq_landrace/K_DARTseq_80kWeighted.npy",
    "genotype_panels/gbs_sawyt/K_GBS_SAWYT.QCfiltered.npy",
    "regulatory_model/enformer_windows.h5",
]


def main() -> None:
    rows = []
    for group, paths in [("required", EXPECTED), ("optional", OPTIONAL)]:
        for rel in paths:
            path = Path(rel)
            rows.append((group, rel, path.exists(), path.stat().st_size if path.exists() else 0))
    print("group\tpath\texists\tsize_bytes")
    for row in rows:
        print("\t".join(map(str, row)))
    missing = [rel for group, rel, exists, _ in rows if group == "required" and not exists]
    if missing:
        raise SystemExit(f"Missing required outputs: {missing}")


if __name__ == "__main__":
    main()
