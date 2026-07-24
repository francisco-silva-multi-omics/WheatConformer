# Frozen Reaction-Norm Outer Evaluation

The completed inner-validation screen selected
`reaction_norm_identity_covariance` using the predeclared acceptance thresholds.
The correlated-trait candidate remains an interpretable diagnostic and is not
eligible for outer evaluation.

`ABOVE_GROUND_BIOMASS` remains in the seven-trait model. Its result must be
reported as exploratory because it did not improve over the matched nonlinear
reference during inner validation. Removing it now would define a new model and
would require a new inner-selection protocol.

The outer protocol fixes:

- the selected candidate and all training settings;
- the exact six-kernel set;
- three deterministic inner-fold members per outer fold;
- all 23 outer folds across five generalization scenarios;
- train-only scaling, weights, trait covariance, centering, and Nystrom factors;
- arithmetic ensembling under the frozen structural-support policy.

The suite first freezes and checksums all completed inner-screen inputs. It then
evaluates unseen environments, unseen genotypes, unseen genotypes and
environments, temporal holdouts, and country holdouts without further model
selection. The final holdout environment manifest is used only as an exclusion
check; its outcomes remain sealed.

Run one outer fold with:

```bash
bash scripts/run_multitrait_reaction_norm_outer_fold.sh \
  . unseen_genotypes 0
```

Run and verify the complete grid with:

```bash
nohup bash scripts/run_multitrait_reaction_norm_outer_suite.sh . \
  > logs/reaction_norm_outer_evaluation_v1.nohup.log 2>&1 &
```

The final summaries are written under
`trained_models/reaction_norm_outer_evaluation_v1_summary`; provenance and
complete-grid checks are written under
`audit/reaction_norm_outer_evaluation_v1`.
