# Environment Kernel Fix

The reproducible pipeline includes the corrected environment kernel workflow.

Instead of one naive kernel from all `EnvData` traits, the environment relationship is split into components:

```text
K_geo       latitude / longitude / altitude
K_weather   temperature / rainfall / radiation / humidity
K_stress    heat days / drought indices / vapor pressure deficit
K_mgmt      sowing date / irrigation / fertilization if available
```

The combined kernel is:

```text
K_E = w_geo K_geo + w_weather K_weather + w_stress K_stress + w_mgmt K_mgmt
```

## Scripts

```text
build_trial_weather_fetch_manifest.py
fetch_nasa_power_trial_weather.py
fetch_openmeteo_trial_weather.py
build_environment_component_kernels.py
build_future_rcp_environment_matrices.py
```

## Historical Environments

To use only local `EnvData` and `Loc_data`:

```bash
python build_environment_component_kernels.py
```

To fetch external weather first:

```bash
python build_trial_weather_fetch_manifest.py
python fetch_nasa_power_trial_weather.py
python fetch_openmeteo_trial_weather.py
python build_environment_component_kernels.py
```

or through the core runner:

```bash
FETCH_WEATHER=1 bash scripts/01_run_core_pipeline.sh
```

## Main Outputs

```text
environment/K_geo.npy
environment/K_weather.npy
environment/K_stress.npy
environment/K_mgmt.npy
environment/K_E.npy
environment/env_kernel_sample_order.tsv
environment/env_kernel_component_weights.tsv
environment/env_kernel_feature_manifest.tsv
environment/env_kernel_coverage_summary.tsv
environment/env_feature_scaling_parameters.tsv
environment/env_features_geo.parquet
environment/env_features_weather.parquet
environment/env_features_stress.parquet
environment/env_features_mgmt.parquet
```

If external weather files are missing, the script falls back to traits available in `EnvData`.

## Future RCP Projection

After historical kernels exist, future RCP weather features can be projected with:

```bash
python build_future_rcp_environment_matrices.py \
  --input environment/future_rcp_weather_features.tsv
```

Expected future input columns include:

```text
future_env_id
base_env_id
scenario / rcp
year / period
weather/stress features matching historical feature names
```

The future workflow reuses historical scaling parameters from:

```text
environment/env_feature_scaling_parameters.tsv
```
