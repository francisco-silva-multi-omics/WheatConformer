# Weather coverage recovery

This workflow recovers missing weather/stress coverage without modifying the current
corrected environment kernels. It writes the manifest, API cache, audits, rebuilt
kernels, climatology expert, registries, and model runs to versioned directories.

The previously reported count of 845 is the missing count for the weather and stress
components. The audited corrected management component had 18 missing environments,
not 845, and is not globally imputed by this workflow.

## Recovery contract

1. Classify every environment before recovery: missing window, missing coordinates,
   dates outside NASA coverage, failed request, or ready but not fetched.
2. Report overlap with all 11,616 environments, the compact pedigree-model order,
   traits, countries, cycles, and each environment-held-out split.
3. Recover dates from the canonical fieldbook-derived environment table and optional
   reviewed supplements. Every inferred window records its source.
4. Match coordinates by normalized `trial_dir + Loc_no`, then stable
   `Country + Loc_no`. Only approved rows from an optional location registry can
   fill remaining coordinates.
5. Retry NASA POWER from the request cache without clamping pre-1981 dates. Use
   Open-Meteo ERA5 for still-missing target environments.
6. Build location-and-season climatology as a separate kernel expert. Never replace
   missing rows with a global weather average.
7. Materialize and certify API/climatology coverage masks. Training multiplies
   trait-level gates by these masks without row-wise renormalization.
8. Adopt recovered kernels only from repeated-seed validation results. Test results
   are reported but are not used by the adoption rule.

## Optional reviewed supplements

The date supplement is TSV or CSV with `env_id` and one or more of
`sowing_date`, `emergence_date`, `harvest_start_date`, and
`harvest_finish_date`. An optional `provenance` column is preserved.

The location registry is TSV or CSV with `latitude`, `longitude`, and
`review_status`. `review_status` must be `approved` or `reviewed`. Rows can be keyed
by `env_id`, or by both `Country` and `Loc_no`.

## Server execution

Run from the repository clone, while keeping the data root separate:

```bash
set +u

CODE="$HOME/tools/WheatConformer"
DATA="/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente"
PYTHON="$HOME/tools/tf_wheat_cpu/bin/python"

git -C "$CODE" fetch origin
git -C "$CODE" switch main
git -C "$CODE" pull --ff-only origin main

cd "$DATA"
mkdir -p logs

nohup env \
  PYTHON="$PYTHON" \
  WHEATCONFORMER_CODE_ROOT="$CODE" \
  WEATHER_RECOVERY_TAG="v1" \
  WEATHER_RECOVERY_TARGET_SCOPE="model" \
  WEATHER_RECOVERY_WORKERS="4" \
  bash "$CODE/scripts/run_weather_recovery_pipeline.sh" "$DATA" \
  > logs/weather_recovery_v1.nohup.log 2>&1 &
```

Set `WEATHER_RECOVERY_TARGET_SCOPE=all` only after model-environment recovery has
been inspected. Optional reviewed files are supplied with
`WEATHER_RECOVERY_DATE_SUPPLEMENT` and `WEATHER_RECOVERY_LOCATION_REGISTRY`.

The default run stops after rebuilding and auditing the isolated kernels. To train
and compare the seven-trait model across seeds 2026-2029, rerun with:

```bash
nohup env \
  PYTHON="$PYTHON" \
  WHEATCONFORMER_CODE_ROOT="$CODE" \
  WEATHER_RECOVERY_TAG="v1" \
  WEATHER_RECOVERY_RUN_TRAINING="1" \
  WEATHER_RECOVERY_BASELINE_VARIANT="<current-corrected-variant>" \
  MULTITRAIT_WEIGHT_POWER="0" \
  MULTITRAIT_INCLUDE_DISABLED_KERNELS="K_E_TGW_V2" \
  bash "$CODE/scripts/run_weather_recovery_pipeline.sh" "$DATA" \
  > logs/weather_recovery_v1_training.nohup.log 2>&1 &
```

The final decision is written to
`trained_models/model_comparisons/weather_recovery_v1_adoption_decision.tsv`.
Acceptance requires a complete comparison grid, four seeds, improved mean
validation normalized RMSE and Pearson, at least 75% seed-level wins for each, and
at least 60% validation seed-trait RMSE wins. Otherwise the decision remains
`retain_current_corrected_kernel`.

## Primary outputs

- `model_kernels/weather_recovery_audit_v1/final/weather_recovery_availability_summary.tsv`
- `model_kernels/weather_recovery_audit_v1/final/weather_recovery_cause_summary.tsv`
- `model_kernels/weather_recovery_audit_v1/final/weather_recovery_by_trait.tsv`
- `model_kernels/weather_recovery_audit_v1/final/weather_recovery_by_country.tsv`
- `model_kernels/weather_recovery_audit_v1/final/weather_recovery_by_cycle.tsv`
- `model_kernels/weather_recovery_audit_v1/final/weather_recovery_by_split.tsv`
- `model_kernels/weather_recovery_audit_v1/final/weather_recovery_split_leakage.tsv`
- `environment_weather_recovery_kernels_v1/K_weather.npy`
- `environment_weather_recovery_kernels_v1/K_stress.npy`
- `environment_weather_recovery_kernels_v1/K_climatology.npy`
- `environment_weather_recovery_kernels_v1/environment_expert_coverage.tsv`

The original `environment/` directory and every prior corrected-kernel directory
remain unchanged.
