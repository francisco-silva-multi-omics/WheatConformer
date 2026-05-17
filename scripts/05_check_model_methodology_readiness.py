from __future__ import annotations

from pathlib import Path


CHECKS = [
    ("canonical_database", "integrated_database/canonical_trial_genotype_environment_plot_table.parquet", "implemented"),
    ("stage1_adjusted_phenotypes", "phenotypes/stage1_adjusted_phenotypes.parquet", "implemented"),
    ("hmp_kernel", "genotype_panels/hmp/K_HMP.QCfiltered.npy", "implemented"),
    ("environment_kernel", "environment/K_E.npy", "implemented"),
    ("environment_geo_kernel", "environment/K_geo.npy", "implemented"),
    ("environment_weather_kernel", "environment/K_weather.npy", "implemented"),
    ("environment_stress_kernel", "environment/K_stress.npy", "implemented"),
    ("environment_management_kernel", "environment/K_mgmt.npy", "implemented"),
    ("stage1_hmp_env_model_inputs", "model_kernels/stage1_hmp_env/stage1_hmp_env_model_ready_stage1_observations.parquet", "implemented"),
    ("tensorflow_multikernel_baseline", "trained_models/stage1_mkl", "run_training_to_generate"),
    ("pedigree_kernel_K_A", "genotype_panels/pedigree/K_A.npy", "missing_methodology_component"),
    ("functional_embedding_kernel_K_z", "model_kernels/K_z.npy", "missing_methodology_component"),
    ("pangenome_graph_gfa", "pangenome_resources/graph/iwgsc_plus_panel.gfa", "missing_external_graph_component"),
    ("pangenome_path_dictionary", "pangenome_resources/graph/genotype_path_dictionary.tsv", "missing_external_graph_component"),
    ("future_rcp_environment_features", "environment/future_rcp_weather_features.tsv", "missing_if_rcp_analysis_required"),
    ("future_rcp_environment_kernel", "environment/future/K_E_future.npy", "missing_if_rcp_analysis_required"),
    ("validation_ablation_report", "trained_models/validation_ablation_report.tsv", "missing_methodology_component"),
]


def main() -> None:
    print("component\tpath\texists\tmethodology_status")
    missing = []
    for component, rel, status in CHECKS:
        exists = Path(rel).exists()
        print(f"{component}\t{rel}\t{exists}\t{status}")
        if not exists and status.startswith("missing"):
            missing.append(component)
    if missing:
        print("\nMissing methodology-level components:")
        for item in missing:
            print(f"- {item}")


if __name__ == "__main__":
    main()
