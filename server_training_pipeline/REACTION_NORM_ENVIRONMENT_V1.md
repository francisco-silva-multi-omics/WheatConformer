# Reaction-Norm Environment Architecture V1

`E_REACTION_NORM_V1` is a phenotype-blind, fold-local environment design matrix for
the multi-trait reaction-norm baseline. It does not replace the certified corrected
generic environment components. The inner screen compares:

1. The current corrected geo/weather/stress/management kernel-product reaction norm.
2. The same main effects plus `K_E_REACTION_NORM_V1` and explicit genotype-by-feature
   slopes from `E_REACTION_NORM_V1`.

`K_E_TGW_V2` remains active for 1000-grain weight in both arms. The DTH, DTM, grain
yield, and plant-height trait kernels remain diagnostic and are not activated.

## Leakage Contract

- Weather windows are fixed relative to sowing. Actual heading or maturity outcomes
  are forbidden as feature-window boundaries.
- Climatology donors, imputation values, scaling, and kernel normalization use only
  the immutable outer-training environments.
- The comparison uses all 15 unseen-genotype inner-validation folds with matched
  seeds and validation observations.
- Outer-test metrics and the final holdout are unavailable during selection.

## Server Run

Update the code clone, then run the complete inner screen:

```bash
set +u

DATA="/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente"
CODE="$HOME/tools/WheatConformer"
PYTHON="$HOME/tools/tf_wheat_cpu/bin/python"

git -C "$CODE" fetch origin audit/forensic-kernel-fixes
git -C "$CODE" checkout --detach <COMMIT_SHA>

cd "$DATA"
nohup env \
  PYTHON="$PYTHON" \
  WHEATCONFORMER_CODE_ROOT="$CODE" \
  bash "$CODE/scripts/run_reaction_norm_environment_inner_screen.sh" "$DATA" all \
  > logs/reaction_norm_environment_inner_screen_v1.nohup.log 2>&1 &

tail -f logs/reaction_norm_environment_inner_screen_v1.nohup.log
```

The runner derives the raw environment and recovered-weather directories from each
fold's certified `K_E.qc.json`. Override them only when deliberately testing another
versioned source:

```bash
REACTION_ENVIRONMENT_INPUT_DIR=/absolute/raw/environment \
REACTION_WEATHER_DIR=/absolute/recovered/weather \
REACTION_WINDOW_FEATURES=/absolute/agronomic_api_weather_windows.tsv \
bash "$CODE/scripts/run_reaction_norm_environment_inner_screen.sh" "$DATA" all
```

## Decision Outputs

The complete run writes:

```text
model_kernels/reaction_norm_environment_inner_screen_v1/summary/unseen_genotypes/
  reaction_norm_environment_screen_summary.tsv
  reaction_norm_environment_screen_trait_summary.tsv
  reaction_norm_environment_screen_paired_metrics.tsv
  selected_reaction_norm_environment_architecture.json
```

The selection lock deliberately records `outer_evaluation_allowed: false`. A new
outer protocol must be generated and bound to the selected architecture and artifact
hashes before the 69 outer member fits are launched. The former outer-v1 runner exits
immediately while this selection is pending.
