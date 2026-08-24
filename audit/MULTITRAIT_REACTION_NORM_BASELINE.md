# Multi-trait reaction-norm baseline

This baseline is an independent statistical comparator to the existing nonlinear
multi-kernel network. It does not modify any certified nested-evaluation output.

## Frozen architecture

The model uses exactly:

- `K_A_CANONICAL_V3` as the additive relationship backbone.
- `K_E_GEO`, `K_E_WEATHER`, `K_E_STRESS`, and `K_E_MGMT` as generic
  environment components.
- `K_E_TGW_V2` for `1000_GRAIN_WEIGHT` only.
- A main effect for every active kernel.
- A reaction-norm interaction between `K_A_CANONICAL_V3` and every eligible
  environment kernel.
- Trait-specific intercepts and Gaussian residual scales.
- Either independent trait effects or a train-estimated, PSD-clipped trait
  covariance shrunk 25% toward the identity.

Interaction features are deterministic random-Maclaurin approximations to the
product kernel `K_A[g_i,g_j] * K_E[e_i,e_j]`. All kernels are centered and
factorized with train-only Nyström bases. Missing weather coverage gates the
corresponding main and interaction effects instead of filling an unavailable
kernel row as observed.

The immutable settings and acceptance thresholds are in
`server_training_pipeline/reaction_norm_protocol_v1.json`.

## Leakage contract

The initial screen uses the frozen `unseen_genotypes` manifests. For every
inner fold, phenotype scaling, precision-weight statistics, trait covariance,
residual scales, kernel centering, and Nyström factors are fit from inner
training rows only. Outer-test and omitted phenotype-derived fields are cleared
before support counting and the rows are removed before preprocessing.

Selection reads inner-validation metrics only. It fits a matched canonical-v3
nonlinear reference from the same exact six-kernel registry, then compares the
two reaction models using identical seeds, validation observations, kernels,
and order hashes. Earlier nonlinear runs that activated optional climatology
are preserved for their original experiment but are not used as this matched
reference. The screen never generates outer-test predictions.

## Server run

Start with outer fold 0:

```bash
nohup env \
  PYTHON="$HOME/tools/tf_wheat_cpu/bin/python" \
  WHEATCONFORMER_CODE_ROOT="$HOME/tools/WheatConformer" \
  bash "$HOME/tools/WheatConformer/scripts/run_multitrait_reaction_norm_inner_screen.sh" \
    /DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente 0 \
  > /DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente/logs/reaction_norm_outer0.nohup.log 2>&1 &
```

After inspecting only the fold-0 inner summary, run all folds. Completed and
checksum-current fits are reused:

```bash
nohup env \
  PYTHON="$HOME/tools/tf_wheat_cpu/bin/python" \
  WHEATCONFORMER_CODE_ROOT="$HOME/tools/WheatConformer" \
  bash "$HOME/tools/WheatConformer/scripts/run_multitrait_reaction_norm_inner_screen.sh" \
    /DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente all \
  > /DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente/logs/reaction_norm_all.nohup.log 2>&1 &
```

The complete decision table is written to
`model_kernels/reaction_norm_inner_screen_v1/summary/unseen_genotypes/reaction_norm_inner_screen_summary.tsv`.
Passing the promotion threshold is required to replace the nonlinear principal
model. A candidate that does not pass remains the interpretable mixed-model
baseline rather than being discarded.
