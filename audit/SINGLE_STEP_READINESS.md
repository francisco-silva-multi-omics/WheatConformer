# Single-Step Relationship Readiness Audit

This audit is the gate before constructing a single-step pedigree-genomic
relationship matrix. It reads pedigree identifiers, `K_A`, the certified HMP
genomic kernel, their sample orders, and the frozen regulatory-eligibility
certification. It does not read phenotype values, outer-test metrics, or final
holdout outcomes, and it does not modify or construct any relationship kernel.
The default source-lineage manifest is
`metadata_outputs/all_trials_genotype_manifest_resolved.tsv`; set
`PEDIGREE_SOURCE_MANIFEST` only when auditing a versioned replacement.

## Server run

```bash
set +u

CODE="$HOME/tools/WheatConformer"
DATA="/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente"
PYTHON="$HOME/tools/tf_wheat_cpu/bin/python"

git -C "$CODE" fetch origin audit/forensic-kernel-fixes
git -C "$CODE" checkout --detach <COMMITTED_SHA>

cd "$DATA"
env \
  PYTHON="$PYTHON" \
  WHEATCONFORMER_CODE_ROOT="$CODE" \
  bash "$CODE/scripts/run_single_step_readiness_audit.sh" "$DATA" \
  > logs/single_step_readiness_v1.log 2>&1
```

Inspect the decision and the review queues:

```bash
OUT="$DATA/model_kernels/single_step_readiness_v1"

cat "$OUT/single_step_readiness_decision.json"
column -t -s $'\t' "$OUT/single_step_readiness_metrics.tsv"
column -t -s $'\t' "$OUT/source_lineage_conflicts.tsv" | head -50
column -t -s $'\t' "$OUT/source_pedigree_child_mismatches.tsv" | head -50
column -t -s $'\t' "$OUT/uncurated_parent_tokens.tsv" | head -50
column -t -s $'\t' "$OUT/K_A_pedigree_order_mismatches.tsv" | head -50
```

## Interpretation

`K_A` can pass shape, symmetry, PSD, and order checks while still representing
unreviewed lineage. Therefore, `BLOCKED` is the expected result for the current
trial-derived pedigree if conflicting cross histories or noncanonical parent
tokens remain.

Supplying a curated alias registry classifies reviewed aliases but does not make
the existing `K_A` canonical. The parent table must be rewritten with reviewed
stable parent GIDs and `K_A` rebuilt in a new versioned directory before the
single-step gate can pass.

Do not resolve conflicting lineages by retaining the first row. Review them and
record the selected parent identities and provenance explicitly.

## Construction gate

Single-step `H` remains prohibited until all blocking reasons are cleared:

- source children have one reviewed lineage;
- children and parents use stable canonical IDs;
- no self-parent, duplicate-parent, or cyclic relationships exist;
- the pedigree-node universe exactly matches the `K_A` order;
- both relationship kernels pass numerical integrity checks;
- `A22` and HMP `G` have sufficient overlap and valid scale alignment;
- the sampled blended genomic block is PSD.

Warnings about shallow lineage, weak relationship concordance, or poor
conditioning must be reported with any subsequent single-step experiment.
