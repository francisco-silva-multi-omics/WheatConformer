# Reaction-Norm RCP Projection Plan

## Status

`E_REACTION_NORM_RCP_V1` is a planned, phenotype-blind extension of the frozen
`E_REACTION_NORM_V1` environment design. Future prediction remains blocked until
the populated covariates pass every fold-local historical range check in
`reaction_norm_rcp_projection_protocol_v1.json`.

The final holdout, outer-test outcomes, and model-selection metrics are not inputs
to this process.

Before future covariates are populated, run the phenotype-blind feature-readiness
audit. It reconciles the fold-specific 551/553-style feature contracts, optionally
compares earlier feature manifests, detects duplicated raw sources, and assigns
block-specific range rules. A `PASS` from this audit means that the inventory is
complete; it does not authorize future matrices or predictions.

```bash
bash scripts/run_reaction_norm_rcp_feature_readiness_audit.sh /path/to/data
```

The next gate is the historical reconstruction audit:

```bash
bash scripts/run_reaction_norm_rcp_historical_reconstruction_audit.sh /path/to/data
```

This audit is also phenotype-blind. It checks the fixed sowing-relative
precipitation replacement on outer-training environments, fits harvest-anchor
season-length fallbacks from outer-training metadata only, audits the units and
undefined period of the two legacy annual-precipitation fields, and writes both a
fold-lineage work queue and a deduplicated daily-request inventory. It does not
fetch daily data, create an RCP matrix, inspect outcomes, or run predictions.

The harvest-relative monthly fields require daily precipitation. The pre-sowing
moisture field requires antecedent precipitation and ET0, declared irrigation, and
optionally soil moisture. Trial-season aggregates cannot substitute for either
request. Annual precipitation remains blocked until daily backcasts adjudicate an
explicit 12-month period. A `PASS` means that the reconstruction inventory and work
queue are internally complete; `historical_replacement_contract_ready` remains the
separate authorization gate.

Fetch the deduplicated daily backcast inventory with a bounded pilot first:

```bash
REACTION_RCP_DAILY_LIMIT=25 \
  bash scripts/run_reaction_norm_rcp_daily_backcast_fetch.sh /path/to/data
```

After the pilot request index is clean, resume the same cache for all pending
requests with `REACTION_RCP_DAILY_LIMIT=0`. The fetch uses Open-Meteo ERA5 with
explicit GMT dates, writes one checksum-addressable Parquet cache per request, and
refuses dates before the frozen 1940 coverage boundary instead of clipping them.
Daily precipitation and ET0 are requested directly. Shallow and deeper soil
moisture are additionally aggregated from hourly values for the antecedent-moisture
requests. Completing this archive still does not authorize RCP covariate population.

Sparse management and binary lineage fields are checked by physical domain,
support, prevalence, and declared scenario policy. Their fold-standardized z-scores
remain diagnostic and are not interchangeable with the hard z gate used for
continuous climate axes. Frozen duplicated columns must be populated identically;
deduplication belongs in a separately selected environment-design version.

## Projection Unit

One row represents:

```text
site x climate model x realization x scenario x period x sowing policy x management policy
```

Climate-model and realization dimensions remain explicit. Features and predictions
must not be averaged before inference. Ensemble means and uncertainty quantiles are
reporting products produced after modelwise predictions exist.

## Matrix Population

1. Create a certified site registry containing latitude, longitude, elevation, and
   the identity of the corresponding historical environment or site.
2. Supply bias-corrected daily climate for each projection unit. Minimum inputs are
   daily minimum and maximum temperature, precipitation, surface solar radiation,
   and humidity information sufficient to calculate VPD.
3. Declare a sowing policy. It may retain a historical day of year, specify a future
   day of year, or use a frozen weather-trigger rule. Observed target heading and
   maturity dates are forbidden.
4. Declare management. Historical management may be held fixed only when that is an
   explicit scenario. Alternative irrigation and fertilizer assumptions require
   separate projection rows.
5. Recompute every fixed sowing-relative weather, stress, development, and radiation
   feature with the same formulas used by `E_REACTION_NORM_V1`.
6. Derive confidence flags from projection provenance. Projected climate is available,
   but it is neither observed API weather nor historical climatology.
7. Materialize one raw matrix with the exact source-feature union recorded in
   `E_REACTION_NORM_RCP_V1_feature_population_plan.tsv`.
8. Apply each outer fold's frozen imputation and scaling parameters independently.
   This yields 23 fold-specific standardized future matrices in the exact historical
   feature order.

## Range Certification

`certify_reaction_norm_rcp_covariates.py` compares each future row against every
outer-training environment reference. It reports raw training minima/maxima,
1st-99th percentile ranges, standardized distances, nonfinite values, and the
fraction of extreme features per environment.

Certification fails when any required source feature is absent, any environment has
nonfinite features, an absolute standardized value exceeds the hard limit, or too
many features exceed the extreme-distance threshold. A failed matrix remains an
audit artifact and cannot be used for projection.

## Model Inputs After Certification

The standardized `E_REACTION_NORM_RCP_V1` rows populate genotype-specific
reaction-norm slopes directly. Environment main effects require fold-specific
future-to-training cross-kernels:

```text
K_future,training = E_future E_training' / p
```

using the historical fold's normalization. Generic geography, weather, stress, and
management experts remain separate and require their own future-to-training
cross-kernels. `K_E_TGW_V2` needs a separately certified future projection before it
can remain active for TGW; it must not be silently copied from the base environment.

The known-environment trial/intercept route is not available for RCP projections.
All RCP rows use the frozen future-compatible reaction-norm route.
