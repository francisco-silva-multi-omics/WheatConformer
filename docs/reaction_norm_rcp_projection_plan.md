# Reaction-Norm RCP Projection Plan

## Status

`E_REACTION_NORM_RCP_V1` is a planned, phenotype-blind extension of the frozen
`E_REACTION_NORM_V1` environment design. Future prediction remains blocked until
the populated covariates pass every fold-local historical range check in
`reaction_norm_rcp_projection_protocol_v1.json`.

The final holdout, outer-test outcomes, and model-selection metrics are not inputs
to this process.

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
