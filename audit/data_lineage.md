# Data Lineage

| Stage | Producer | Inputs | Outputs | Keys | Transformations |
|---|---|---|---|---|---|
| raw_trials | `build_requested_outputs.py::build_phenotypes_and_environment` | TRIALS_AND_NURSERIES_DATA/** | phenotypes/model_input_phenotypes.tsv | trial/cycle/occ/GID/trait | mean duplicate numeric summaries |
| trial_manifest | `resolve_all_trial_gids.py::main` | TRIALS_AND_NURSERIES_DATA/**/FieldBook*; DOI tables | metadata_outputs/all_trials_genotype_manifest_resolved.tsv | trial/cycle/occ/CID/SID/GID | first/explicit source rules |
| canonical | `build_canonical_integrated_database.py::main` | phenotypes/model_input_phenotypes.tsv; manifests; panel orders | integrated_database/canonical_trial_genotype_environment_plot_table.parquet | trial_key/cycle/occ/resolved_gid; env_id | duplicate phenotype means retained from upstream |
| stage1 | `build_stage1_adjusted_phenotypes.py::main` | canonical parquet; phenotypes/all_rawdata.tsv | phenotypes/stage1_adjusted_phenotypes.parquet | trait/trial/environment/genotype | linear-model adjusted y_tilde; fallback mean |
| K_G | `build_requested_outputs.py::compute_hmp_qc; vanraden_kernel` | HMP HapMap marker file | genotype_panels/hmp/K_HMP.QCfiltered.meanDiag1.npy | sample_id order | marker-mean imputation; VanRaden; mean-diagonal scale |
| K_A | `build_pedigree_kernel.py::build_parent_table; additive_relationship` | trial-derived pedigree manifest | genotype_panels/pedigree/K_A.npy | sample_id order | unknown parents as founders; cycles broken as founders |
| K_E | `build_environment_component_kernels.py::build_env_trait_matrix; standardized_kernel` | environment/envdata.tsv; locdata.tsv; weather API tables | environment/K_E.npy | env_id order | mean imputation; z-score; equal component weights; mean-diagonal scale |
| stage1_compact | `build_stage1_model_kernels.py::main` | stage1 phenotype; K_G/K_A; K_E | model_kernels/*/*_K_[GE]_unique.npy | sample_id/env_id | compact subsetting |
| K_GxE | `build_stage1_model_kernels.py::main` | observation indices; compact K_G/K_E | *_K_GE_hadamard.npy | observation_index | Hadamard K_G[g,g'] * K_E[e,e'] |
| multitrait_ledger | `server_training_pipeline/build_multitrait_ledger.py::main` | stage1 pedigree observations; kernel registry | model_kernels/multitrait_*/ledger.parquet | observation/genotype/environment/trait | weight tempering |
| kernel_registry | `server_training_pipeline/prepare_multitrait_kernel_registry.py::main` | K_A; HMP/GBS K_G; environment components; trait K_E | model_kernels/multitrait_*/kernel_registry.tsv | explicit target orders | masked aligned experts |
| split | `server_training_pipeline/split_utils.py::make_split` | multitrait ledger | train/validation/test indices | declared grouping column | deterministic RNG seed |
| training | `server_training_pipeline/train_multitrait_multikernel_tf.py::main` | ledger; registry; split | trained_models/** | ledger row order | low-rank factors; multitrait heads |
