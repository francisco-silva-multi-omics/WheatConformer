#!/usr/bin/env bash
set -euo pipefail

# Recompute every non-pangenome input needed by the methodology:
# phenotype harmonization, environment kernels, genotype kernels/catalogs,
# stage-1 adjusted phenotypes, compact model matrices, optional pedigree,
# optional regulatory K_z, optional RCP matrices, then matrix-readiness checks.
#
# This script intentionally does not call Minigraph-Cactus, cactus-pangenome,
# minigraph, vg, odgi, or any pangenome graph builder. The graph pangenome is
# treated as an external artifact, preferably Zenodo record 6085239.

ROOT="${1:-$(pwd)}"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
LOG_DIR="${LOG_DIR:-logs/non_pangenome_recompute_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR" model_kernels

FETCH_WEATHER="${FETCH_WEATHER:-0}"
RUN_GBS="${RUN_GBS:-1}"
RUN_80K_PRIORS="${RUN_80K_PRIORS:-1}"
RUN_GERMPLASM_RESOLVER="${RUN_GERMPLASM_RESOLVER:-1}"
RUN_PEDIGREE="${RUN_PEDIGREE:-1}"
RUN_RCP="${RUN_RCP:-1}"
RUN_MULTIOMICS_MANIFEST="${RUN_MULTIOMICS_MANIFEST:-1}"
RUN_KZ="${RUN_KZ:-1}"
STAGE1_CHUNKSIZE="${STAGE1_CHUNKSIZE:-250000}"
RAW_CHUNKSIZE="${RAW_CHUNKSIZE:-250000}"
TRAIT_REGEX="${TRAIT_REGEX:-}"
ZENODO_PANGENOME_DIR="${ZENODO_PANGENOME_DIR:-pangenome_resources/graph}"
PANGENOME_GFA="${PANGENOME_GFA:-${ZENODO_PANGENOME_DIR}/15-wheat10+.gfa.gz}"
PANGENOME_BED="${PANGENOME_BED:-${ZENODO_PANGENOME_DIR}/15-wheat10+.bed.gz}"
PANGENOME_GBZ="${PANGENOME_GBZ:-${ZENODO_PANGENOME_DIR}/index.giraffe.gbz}"
PANGENOME_MIN="${PANGENOME_MIN:-${ZENODO_PANGENOME_DIR}/index.min}"
PANGENOME_DIST="${PANGENOME_DIST:-${ZENODO_PANGENOME_DIR}/index.dist}"
PANGENOME_HAL="${PANGENOME_HAL:-}"
PEDIGREE_TABLE="${PEDIGREE_TABLE:-}"
PEDIGREE_MANIFEST="${PEDIGREE_MANIFEST:-metadata_outputs/all_trials_genotype_manifest_resolved.tsv}"
EXTERNAL_GERMPLASM_TABLES="${EXTERNAL_GERMPLASM_TABLES:-}"
MULTIOMICS_DIR="${MULTIOMICS_DIR:-multi_omics_data}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

run_step() {
  local name="$1"
  shift
  log "START ${name}"
  "$@" >"${LOG_DIR}/${name}.stdout.log" 2>"${LOG_DIR}/${name}.stderr.log"
  log "DONE  ${name}"
}

run_optional_step() {
  local name="$1"
  shift
  log "START optional ${name}"
  if "$@" >"${LOG_DIR}/${name}.stdout.log" 2>"${LOG_DIR}/${name}.stderr.log"; then
    log "DONE  optional ${name}"
  else
    log "WARN  optional ${name} failed; see ${LOG_DIR}/${name}.stderr.log"
  fi
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -s "$path" ]]; then
    log "ERROR missing ${label}: ${path}"
    exit 2
  fi
}

log "Root: ${ROOT}"
log "Logs: ${LOG_DIR}"
log "Pangenome is external only. GFA='${PANGENOME_GFA}' BED='${PANGENOME_BED}' GBZ='${PANGENOME_GBZ}' MIN='${PANGENOME_MIN}' DIST='${PANGENOME_DIST}' HAL='${PANGENOME_HAL}'"

run_step 01_trial_gid_map "$PYTHON" trial_GID_map.py
run_step 02_requested_outputs "$PYTHON" build_baseline.py
run_step 02b_gaussian_genomic_kernel "$PYTHON" build_gaussian_genomic_kernel.py \
  --linear-kernel genotype_panels/hmp/K_HMP.QCfiltered.meanDiag1.npy \
  --sample-order genotype_panels/hmp/hmp_K_sample_order.QCfiltered.tsv \
  --out-kernel genotype_panels/hmp/K_HMP.QCfiltered.gaussian.npy \
  --out-qc genotype_panels/hmp/K_HMP.QCfiltered.gaussian.qc.json \
  --gamma-multiplier "${GAUSSIAN_GAMMA_MULTIPLIER:-1.0}" \
  --median-sample-size "${GAUSSIAN_MEDIAN_SAMPLE_SIZE:-2048}" \
  --chunk-size "${GAUSSIAN_CHUNK_SIZE:-256}"
run_step 03_next_integration_layer "$PYTHON" build_next_integration_layer.py
run_step 04_dartseq_landrace_qc "$PYTHON" build_dartseq_landrace_diversity_qc.py
run_step 05_integrate_80k_catalog "$PYTHON" integrate_80k_diversity_panel.py

if [[ "$RUN_GBS" == "1" && -d GBS ]]; then
  run_optional_step 06_gbs_sawyt_panel "$PYTHON" build_gbs_sawyt_panel.py --gbs-dir GBS --out-dir genotype_panels/gbs_sawyt
else
  log "SKIP 06_gbs_sawyt_panel: RUN_GBS=${RUN_GBS}, GBS dir present=$([[ -d GBS ]] && echo yes || echo no)"
fi

if [[ "$FETCH_WEATHER" == "1" ]]; then
  run_step 07a_weather_manifest "$PYTHON" build_trial_weather_fetch_manifest.py
  run_optional_step 07b_fetch_nasa_power "$PYTHON" fetch_nasa_power_trial_weather.py
  run_optional_step 07c_fetch_openmeteo "$PYTHON" fetch_openmeteo_trial_weather.py
else
  log "SKIP weather downloads: FETCH_WEATHER=0; using existing fetched weather tables plus EnvData/Loc_data fallback"
fi
run_step 07_environment_component_kernels "$PYTHON" build_environment_component_kernels.py

if [[ "$RUN_80K_PRIORS" == "1" && -d 80k ]]; then
  run_step 08a_80k_marker_priors "$PYTHON" server_80k_pipeline/build_80k_marker_priors.py \
    --input-dir 80k \
    --out-dir genotype_panels/diversity_80k \
    --chunksize "${MARKER_PRIOR_CHUNKSIZE:-1000}" \
    --write-fasta \
    --fasta-overlap-only

  if [[ -s genotype_panels/dartseq_landrace/dartseq_landrace_marker_by_sample.parquet && -s genotype_panels/diversity_80k/diversity_80k_marker_prior_features.parquet ]]; then
    run_step 08b_80k_weighted_dartseq_kernel "$PYTHON" server_80k_pipeline/build_80k_weighted_kernel.py \
      --genotype-matrix genotype_panels/dartseq_landrace/dartseq_landrace_marker_by_sample.parquet \
      --prior-table genotype_panels/diversity_80k/diversity_80k_marker_prior_features.parquet \
      --out-dir genotype_panels/dartseq_landrace \
      --prefix K_DARTseq_80kWeighted \
      --orientation marker_by_sample \
      --marker-col marker_id
  else
    log "SKIP 08b_80k_weighted_dartseq_kernel: missing DArTseq matrix or 80k prior table"
  fi
else
  log "SKIP 08_80k_priors: RUN_80K_PRIORS=${RUN_80K_PRIORS}, 80k dir present=$([[ -d 80k ]] && echo yes || echo no)"
fi

run_step 09_canonical_integrated_database "$PYTHON" build_canonical_integrated_database.py \
  --raw-chunksize "$RAW_CHUNKSIZE" \
  --write-tsv

stage1_args=(build_stage1_adjusted_phenotypes.py --chunksize "$STAGE1_CHUNKSIZE" --include-plot-linear --write-tsv)
if [[ -n "$TRAIT_REGEX" ]]; then
  log "NOTE TRAIT_REGEX is recorded for downstream model filtering, but stage-1 script accepts repeated exact --trait values."
fi
run_step 10_stage1_adjusted_phenotypes "$PYTHON" "${stage1_args[@]}"

if [[ -s phenotypes/stage1_adjusted_phenotypes.parquet ]]; then
  STAGE1_PHENOTYPES="phenotypes/stage1_adjusted_phenotypes.parquet"
elif [[ -s phenotypes/stage1_adjusted_phenotypes.tsv.gz ]]; then
  STAGE1_PHENOTYPES="phenotypes/stage1_adjusted_phenotypes.tsv.gz"
else
  require_file phenotypes/stage1_adjusted_phenotypes.parquet "stage-1 adjusted phenotype parquet or TSV fallback"
fi
log "Stage-1 phenotype input for model matrices: ${STAGE1_PHENOTYPES}"
require_file genotype_panels/hmp/K_HMP.QCfiltered.npy "HMP QC genotype kernel"
require_file genotype_panels/hmp/K_HMP.QCfiltered.meanDiag1.npy "mean-diagonal-scaled HMP QC genotype kernel"
require_file genotype_panels/hmp/K_HMP.QCfiltered.gaussian.npy "Gaussian HMP genomic kernel"
require_file genotype_panels/hmp/hmp_K_sample_order.QCfiltered.tsv "HMP QC sample order"
require_file environment/K_E.npy "environment kernel"
require_file environment/env_kernel_sample_order.tsv "environment kernel order"
HMP_KERNEL="genotype_panels/hmp/K_HMP.QCfiltered.meanDiag1.npy"

hmp_model_args=(
  build_stage1_model_kernels.py
  --stage1-phenotypes "$STAGE1_PHENOTYPES"
  --geno-kernel "$HMP_KERNEL"
  --geno-rbf-kernel genotype_panels/hmp/K_HMP.QCfiltered.gaussian.npy
  --require-geno-rbf
  --geno-order genotype_panels/hmp/hmp_K_sample_order.QCfiltered.tsv
  --env-kernel environment/K_E.npy
  --env-order environment/env_kernel_sample_order.tsv
  --out-dir model_kernels/stage1_hmp_env
  --prefix stage1_hmp_env
  --write-tsv
)
if [[ -n "$TRAIT_REGEX" ]]; then
  hmp_model_args+=(--trait-regex "$TRAIT_REGEX")
fi
run_step 11_hmp_stage1_model_inputs "$PYTHON" "${hmp_model_args[@]}"

if [[ "$RUN_GBS" == "1" && -s genotype_panels/gbs_sawyt/K_GBS_SAWYT.QCfiltered.npy ]]; then
  gbs_model_args=(
    build_stage1_model_kernels.py
    --stage1-phenotypes "$STAGE1_PHENOTYPES"
    --geno-kernel genotype_panels/gbs_sawyt/K_GBS_SAWYT.QCfiltered.npy
    --geno-order genotype_panels/gbs_sawyt/gbs_sawyt_K_sample_order.QCfiltered.tsv
    --env-kernel environment/K_E.npy
    --env-order environment/env_kernel_sample_order.tsv
    --out-dir model_kernels/stage1_gbs_sawyt_env
    --prefix stage1_gbs_sawyt_env
    --write-tsv
  )
  if [[ -n "$TRAIT_REGEX" ]]; then
    gbs_model_args+=(--trait-regex "$TRAIT_REGEX")
  fi
  run_optional_step 12_gbs_stage1_model_inputs "$PYTHON" "${gbs_model_args[@]}"
else
  log "SKIP 12_gbs_stage1_model_inputs: missing GBS QC kernel"
fi

if [[ "$RUN_GERMPLASM_RESOLVER" == "1" && -s "$PEDIGREE_MANIFEST" ]]; then
  resolver_args=(
    build_cross_germplasm_resolver.py
    --root "$ROOT"
    --manifest "$PEDIGREE_MANIFEST"
    --stage1-phenotypes "$STAGE1_PHENOTYPES"
    --out-dir genotype_panels/germplasm_resolver
  )
  if [[ -n "$EXTERNAL_GERMPLASM_TABLES" ]]; then
    IFS=':' read -r -a external_tables <<< "$EXTERNAL_GERMPLASM_TABLES"
    for table in "${external_tables[@]}"; do
      if [[ -s "$table" ]]; then
        resolver_args+=(--external-table "$table")
      else
        log "WARN external germplasm table missing or empty: $table"
      fi
    done
  fi
  run_optional_step 12b_cross_germplasm_resolver "$PYTHON" "${resolver_args[@]}"
else
  log "SKIP 12b_cross_germplasm_resolver: RUN_GERMPLASM_RESOLVER=${RUN_GERMPLASM_RESOLVER}, manifest present=$([[ -s "$PEDIGREE_MANIFEST" ]] && echo yes || echo no)"
fi

if [[ "$RUN_PEDIGREE" == "1" ]]; then
  if [[ -z "$PEDIGREE_TABLE" && -s "$PEDIGREE_MANIFEST" ]]; then
    PEDIGREE_TABLE="genotype_panels/pedigree/trial_derived_pedigree_table.tsv"
    run_optional_step 13a_extract_trial_pedigree "$PYTHON" extract_trial_pedigree_from_manifest.py \
      --manifest "$PEDIGREE_MANIFEST" \
      --out-table "$PEDIGREE_TABLE" \
      --out-qc genotype_panels/pedigree/trial_derived_pedigree_qc.tsv
  fi

  if [[ -n "$PEDIGREE_TABLE" && -s "$PEDIGREE_TABLE" ]]; then
  run_optional_step 13_pedigree_kernel "$PYTHON" build_pedigree_kernel.py \
    --pedigree-table "$PEDIGREE_TABLE" \
    --id-col sample_id \
    --cross-col cross_name \
    --out-dir genotype_panels/pedigree \
    --prefix K_A \
    --scale-mean-diagonal
  else
    log "SKIP 13_pedigree_kernel: no PEDIGREE_TABLE and missing PEDIGREE_MANIFEST=${PEDIGREE_MANIFEST}"
  fi
else
  log "SKIP 13_pedigree_kernel: RUN_PEDIGREE=${RUN_PEDIGREE}"
fi

if [[ "$RUN_MULTIOMICS_MANIFEST" == "1" && -d "$MULTIOMICS_DIR" ]]; then
  run_optional_step 14_multiomics_manifest "$PYTHON" server_training_pipeline/build_multiomics_manifest.py \
    --omics-dir "$MULTIOMICS_DIR" \
    --out functional_annotation/multiomics_file_manifest.tsv
else
  log "SKIP 14_multiomics_manifest: MULTIOMICS_DIR='${MULTIOMICS_DIR}' not present or disabled"
fi

if [[ "$RUN_KZ" == "1" && -s regulatory_model/embeddings/hmp_regulatory_genotype_regulatory_embeddings.npy && -s regulatory_model/embeddings/hmp_regulatory_genotype_regulatory_embedding_order.tsv ]]; then
  run_optional_step 15_regulatory_Kz "$PYTHON" server_training_pipeline/build_Kz_from_embeddings.py \
    --embedding-npy regulatory_model/embeddings/hmp_regulatory_genotype_regulatory_embeddings.npy \
    --order regulatory_model/embeddings/hmp_regulatory_genotype_regulatory_embedding_order.tsv \
    --id-col sample_id \
    --out-dir model_kernels \
    --prefix K_z \
    --kernel linear \
    --pca-components "${KZ_PCA_COMPONENTS:-128}"
else
  log "SKIP 15_regulatory_Kz: embeddings absent; train/extract regulatory embeddings first if K_z is required"
fi

if [[ "$RUN_RCP" == "1" ]]; then
  run_optional_step 16_future_rcp_environment "$PYTHON" build_future_rcp_environment_matrices.py --allow-partial-components
else
  log "SKIP 16_future_rcp_environment: RUN_RCP=0"
fi

validate_args=(
  validate_model_input_matrices.py
  --root "$ROOT"
  --out-dir model_kernels/readiness
)
if [[ -n "$PANGENOME_GFA" ]]; then
  validate_args+=(--pangenome-output "$PANGENOME_GFA")
fi
if [[ -n "$PANGENOME_BED" ]]; then
  validate_args+=(--pangenome-output "$PANGENOME_BED")
fi
if [[ -n "$PANGENOME_GBZ" ]]; then
  validate_args+=(--pangenome-output "$PANGENOME_GBZ")
fi
if [[ -n "$PANGENOME_MIN" ]]; then
  validate_args+=(--pangenome-output "$PANGENOME_MIN")
fi
if [[ -n "$PANGENOME_DIST" ]]; then
  validate_args+=(--pangenome-output "$PANGENOME_DIST")
fi
if [[ -n "$PANGENOME_HAL" ]]; then
  validate_args+=(--pangenome-output "$PANGENOME_HAL")
fi
run_step 17_validate_model_input_matrices "$PYTHON" "${validate_args[@]}"

log "Non-pangenome recompute complete."
log "Readiness report: model_kernels/readiness/model_input_readiness_report.tsv"
log "Training manifest: model_kernels/readiness/model_input_manifest.json"
