#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

echo "[1/9] Build/refresh trial GID metadata"
python trial_GID_map.py

echo "[2/9] Build requested genotype, phenotype, environment, annotation outputs"
python build_baseline.py

echo "[2b/9] Build Gaussian/RBF genomic kernel"
python build_gaussian_genomic_kernel.py \
  --gamma-multiplier "${GAUSSIAN_GAMMA_MULTIPLIER:-1.0}"

echo "[3/9] Build harmonized phenotype/MAS/functional integration layer"
python build_next_integration_layer.py

echo "[4/9] DArTseq landrace external diversity QC"
python build_dartseq_landrace_diversity_qc.py

echo "[5/9] 80k external diversity catalog"
python integrate_80k_diversity_panel.py

echo "[6/9] GBS SAWYT panel"
python build_gbs_sawyt_panel.py --gbs-dir GBS --out-dir genotype_panels/gbs_sawyt || echo "GBS step skipped or failed; see logs/stdout"

echo "[7/9] Environment component kernels"
if [[ "${FETCH_WEATHER:-0}" == "1" ]]; then
  echo "  FETCH_WEATHER=1: building weather manifest and fetching NASA POWER/Open-Meteo features"
  python build_trial_weather_fetch_manifest.py
  python fetch_nasa_power_trial_weather.py || echo "  NASA POWER fetch failed or partially failed; continuing with available features"
  python fetch_openmeteo_trial_weather.py || echo "  Open-Meteo fetch failed or partially failed; continuing with available features"
else
  echo "  FETCH_WEATHER not set; using existing weather feature files if present, otherwise EnvData/Loc_data fallback"
fi
python build_environment_component_kernels.py

echo "[8/9] Canonical integrated database"
python build_canonical_integrated_database.py --write-tsv

echo "[9/9] Stage-1 adjusted phenotypes"
python build_stage1_adjusted_phenotypes.py --chunksize 250000 --include-plot-linear --write-tsv

echo "Core pipeline complete."
