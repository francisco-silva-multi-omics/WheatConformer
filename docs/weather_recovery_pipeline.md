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
3. Recover dates from the canonical fieldbook-derived environment table, an optional
   provenance-aware scan of raw `EnvData` files, and optional reviewed supplements.
   Every inferred window records its source.
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

Set `WEATHER_RECOVERY_DISCOVER_RAW_DATES=1` to scan the reorganized raw trial tree.
The scanner accepts a date automatically only when the environment match is exact
or normalization-only, the complete date is unique, and its year is compatible with
the trial cycle. Partial values such as `00/05/1998`, conflicting dates, fuzzy
location matches, and cycle mismatches are retained for review but never converted
into an exact weather window.

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
  WEATHER_RECOVERY_TAG="v2_raw_dates" \
  WEATHER_RECOVERY_TARGET_SCOPE="model" \
  WEATHER_RECOVERY_DISCOVER_RAW_DATES="1" \
  WEATHER_RECOVERY_TRIAL_ROOT="$DATA/TRIALS_AND_NURSERIES" \
  WEATHER_RECOVERY_WORKERS="4" \
  bash "$CODE/scripts/run_weather_recovery_pipeline.sh" "$DATA" \
  > logs/weather_recovery_v2_raw_dates.nohup.log 2>&1 &
```

The audit reports both scopes separately. Even with
`WEATHER_RECOVERY_TARGET_SCOPE=model`, API calls include every retryable
current-manifest request so non-model coverage cannot regress because of stale cached
windows; non-retryable date/coordinate recovery remains focused in the model report.
Optional reviewed files are supplied with
`WEATHER_RECOVERY_DATE_SUPPLEMENT` and `WEATHER_RECOVERY_LOCATION_REGISTRY`.

The default run stops after rebuilding and auditing the isolated kernels. To train
and compare the seven-trait model across seeds 2026-2029, rerun with:

```bash
nohup env \
  PYTHON="$PYTHON" \
  WHEATCONFORMER_CODE_ROOT="$CODE" \
  WEATHER_RECOVERY_TAG="v2_raw_dates" \
  WEATHER_RECOVERY_RUN_TRAINING="1" \
  WEATHER_RECOVERY_BASELINE_VARIANT="<current-corrected-variant>" \
  MULTITRAIT_WEIGHT_POWER="0" \
  MULTITRAIT_INCLUDE_DISABLED_KERNELS="K_E_TGW_V2" \
  bash "$CODE/scripts/run_weather_recovery_pipeline.sh" "$DATA" \
  > logs/weather_recovery_v2_raw_dates_training.nohup.log 2>&1 &
```

The final decision is written to
`trained_models/model_comparisons/weather_recovery_v2_raw_dates_adoption_decision.tsv`.
Acceptance requires a complete comparison grid, four seeds, improved mean
validation normalized RMSE and Pearson, at least 75% seed-level wins for each, and
at least 60% validation seed-trait RMSE wins. Otherwise the decision remains
`retain_current_corrected_kernel`.

### Matched v1 versus v2 experiment

When the v1 model grid has not been trained, use the matched runner instead of
running v2 alone. It builds or resumes all 12 v1 runs and all 12 v2 runs
(three modes by four seeds), recomputes trait support from the frozen ledger using
the training rule of at least 100 training rows and 20 rows in both validation and
test, verifies that run metadata contains exactly that seed-specific retained set,
checks that metrics exactly cover the split/trait pairs present in each prediction
ledger, and requires v1 and v2 to have identical evaluable pairs. A trait that does
not meet the per-seed support rule is reported as support-filtered rather than
assigned a fabricated metric. The runner only then produces the validation-only
adoption decision:

Each newly trained run writes `*_trait_split_support.tsv` and records requested,
retained, and support-filtered traits in `*_run_metadata.json`, including the exact
row thresholds. This makes seed-specific omissions auditable without interpreting
an absent metric as a failed model.

```bash
set +u

CODE="$HOME/tools/WheatConformer"
DATA="/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente"
PYTHON="$HOME/tools/tf_wheat_cpu/bin/python"

cd "$DATA"
mkdir -p logs

nohup env \
  PYTHON="$PYTHON" \
  WHEATCONFORMER_CODE_ROOT="$CODE" \
  WEATHER_RECOVERY_TRIAL_ROOT="$DATA/TRIALS_AND_NURSERIES" \
  WEATHER_RECOVERY_WORKERS="4" \
  MATCHED_WEATHER_FORCE="0" \
  bash "$CODE/scripts/run_matched_weather_recovery_comparison.sh" "$DATA" \
  > logs/matched_weather_recovery_v1_vs_v2.nohup.log 2>&1 &
```

The runner is sequential to avoid competing for server RAM and CPU. Existing
complete runs are reused. Set `MATCHED_WEATHER_FORCE=1` only when an existing run
is known to be stale or incomplete. The comparison contract must contain 12
`PASS` rows and the runner must end with `DONE matched v1 versus v2 raw-date
weather comparison`.

If training completed but the final comparison reports that
`training_configuration` is missing from both variants, do not retrain. Use
`scripts/finalize_matched_weather_recovery_comparison.sh`. The finalizer enforces
safe Python module resolution from the repository clone, backs up original metadata,
fills only an absent configuration from the frozen runner contract, records the
certifying commit and configuration hash, and runs the comparison directly. It
refuses to overwrite any nonempty conflicting configuration.

## Primary outputs

- `model_kernels/weather_recovery_audit_v2_raw_dates/final/weather_recovery_availability_summary.tsv`
- `model_kernels/weather_recovery_audit_v2_raw_dates/final/weather_recovery_cause_summary.tsv`
- `model_kernels/weather_recovery_audit_v2_raw_dates/final/weather_recovery_by_trait.tsv`
- `model_kernels/weather_recovery_audit_v2_raw_dates/final/weather_recovery_by_country.tsv`
- `model_kernels/weather_recovery_audit_v2_raw_dates/final/weather_recovery_by_cycle.tsv`
- `model_kernels/weather_recovery_audit_v2_raw_dates/final/weather_recovery_by_split.tsv`
- `model_kernels/weather_recovery_audit_v2_raw_dates/final/weather_recovery_split_leakage.tsv`
- `environment_weather_recovery_kernels_v2_raw_dates/K_weather.npy`
- `environment_weather_recovery_kernels_v2_raw_dates/K_stress.npy`
- `environment_weather_recovery_kernels_v2_raw_dates/K_climatology.npy`
- `environment_weather_recovery_kernels_v2_raw_dates/environment_expert_coverage.tsv`
- `model_kernels/weather_date_recovery_v2_raw_dates/raw_trial_date_recovery_qc.tsv`
- `model_kernels/weather_date_recovery_v2_raw_dates/raw_trial_date_resolution.tsv`
- `model_kernels/weather_date_recovery_v2_raw_dates/raw_trial_date_conflicts.tsv`
- `model_kernels/weather_date_recovery_v2_raw_dates/weather_date_supplement.tsv`

The original `environment/` directory and every prior corrected-kernel directory
remain unchanged.
