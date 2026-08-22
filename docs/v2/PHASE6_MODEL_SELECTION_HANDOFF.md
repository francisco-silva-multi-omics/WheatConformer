# Stage-1 v2 Phase-6 Model-Selection Handoff

## Scope

This handoff resolves the authoritative Stage-1-v2 inputs before any Phase-6
performance metric is inspected. It binds Phase 5, the parity extension, the
150-state K_A extension, regulatory eligibility, projection-core releases,
CIMMYT pre-QC recovery and H_SEEDS to one code/runtime/selection contract.

## Genetic components

- K_A is the required genetic backbone in every state.
- Seeds DArTseq is available through training-local marker parameters.
- H_SEEDS is an on-demand single-step precision correction over 1,514 accepted
  GIDs shared with K_A. It uses 95% aligned G and 5% A22.
- The H_SEEDS correction is active in 137/150 states. It is masked in 13
  temporal states with fewer than 20 training overlaps; K_A remains active.
- The newly recovered CIMMYT pre-QC calls are an individual masked candidate,
  not a merged marker matrix.
- K_z and unresolved EYT/80K/targeted panels are deferred before metrics.

## Environment components

Historical candidates may use the certified parity components. Projection-
compatible candidates use the exact split-bound E_PROJECTION_CORE_V1 schema:
153 features, rank 64 and training-only transformation/factorization in each of
150 states. All 814 inactive environments remain in the observation population
with explicit masks and mandatory reporting.

## Selection

Phase 1 uses outer fold 1's five inner folds and matched seeds. Advancing
architectures are confirmed over all 125 inner states. Macro trait-by-scenario
normalized RMSE is primary; macro Pearson is the tiebreaker. Calibration,
primary traits, within-environment ranking, information classes and inactive
environments are hard guards. The marker-combination candidate is eligible only
after both individual marker candidates pass.

Outer partitions may be opened once only after the inner decision, code,
historical specification and projection-compatible specification are frozen.
The final holdout remains sealed.

## Runtime

Training and the complete release test suite use the frozen WSL2 Debian runtime:
Python 3.11.15, TensorFlow 2.15.1 and pandas 2.2.3. The Windows `.audit-venv`
remains an audit runtime and is not the certified training environment.

The complete repository suite passed 782 tests in that WSL runtime. The
aggregate handoff is frozen at
`audit/v2/phase6_model_selection_handoff_v1/` with status
`PASS_READY_FOR_STAGE1_V2_PHASE6_INNER_MODEL_SELECTION`.
