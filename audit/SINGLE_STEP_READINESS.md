# Single-Step Relationship Readiness Audit

This audit is the gate before constructing a single-step pedigree-genomic
relationship matrix. It reads pedigree identifiers, `K_A`, the certified HMP
genomic kernel, their sample orders, and the frozen regulatory-eligibility
certification. It does not read phenotype values, outer-test metrics, or final
holdout outcomes, and it does not modify or construct any relationship kernel.
The default source-lineage manifest is
`metadata_outputs/all_trials_genotype_manifest_resolved.tsv`; set
`PEDIGREE_SOURCE_MANIFEST` only when auditing a versioned replacement.

## Canonical pedigree correction

The legacy trial-derived `K_A` split pedigree text at the first convenient
delimiter. That is not valid for compound Purdy/CIMMYT notation such as
`A/B//C/3/D`, where `//` and `/3/` encode cross order. Before auditing or
constructing single-step `H`, build the isolated canonical pedigree:

```bash
set +u

CODE="$HOME/tools/WheatConformer"
DATA="/DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente"
PYTHON="$HOME/tools/tf_wheat_cpu/bin/python"

cd "$DATA"
env \
  PYTHON="$PYTHON" \
  WHEATCONFORMER_CODE_ROOT="$CODE" \
  CANONICAL_PEDIGREE_ALLOW_CONSERVATIVE_FOUNDER_FALLBACK=1 \
  bash "$CODE/scripts/build_canonical_pedigree_v2.sh" "$DATA"
```

This produces, without modifying the legacy pedigree:

- `canonical_parent_registry.tsv`;
- `child_lineage_resolution.tsv`;
- `selfing_review.tsv`;
- `canonical_pedigree_parent_table.tsv`;
- `K_A_CANONICAL_V2.npy` and certified order files;
- a manual decision template for conflicts and selfing records.

Stable `PEDF_*` nodes identify exact named founder expressions and `PEDX_*`
nodes identify deterministic cross subtrees. They are local lineage identities,
not claims of verified global germplasm GIDs. Competing child lineages and
unreviewed selfing relationships are converted to recorded founders when the
conservative fallback is explicitly enabled. No phenotype observation is
removed. A reviewed manual decision file can later replace those fallbacks in a
new version.

The parser follows the cross-order and backcross-dose semantics of the
Purdy/CIMMYT notation. It never treats every slash as an interchangeable text
separator.

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
  SINGLE_STEP_READINESS_OUT_DIR="model_kernels/single_step_readiness_v2" \
  SINGLE_STEP_PEDIGREE_PARENT_TABLE="genotype_panels/pedigree_canonical_v2/canonical_pedigree_parent_table.tsv" \
  SINGLE_STEP_K_A="genotype_panels/pedigree_canonical_v2/K_A_CANONICAL_V2.npy" \
  SINGLE_STEP_K_A_ORDER="genotype_panels/pedigree_canonical_v2/K_A_CANONICAL_V2_sample_order.tsv" \
  SINGLE_STEP_CHILD_ID_REGEX='^(GID[0-9]+|PED[FX]_[A-F0-9]{16})$' \
  SINGLE_STEP_PARENT_ID_REGEX='^(GID[0-9]+|PED[FX]_[A-F0-9]{16})$' \
  STABLE_PARENT_REGISTRY="genotype_panels/pedigree_canonical_v2/canonical_parent_registry.tsv" \
  PEDIGREE_LINEAGE_RESOLUTION="genotype_panels/pedigree_canonical_v2/child_lineage_resolution.tsv" \
  bash "$CODE/scripts/run_single_step_readiness_audit.sh" "$DATA" \
  > logs/single_step_readiness_v2.log 2>&1
```

Inspect the decision and the review queues:

```bash
OUT="$DATA/model_kernels/single_step_readiness_v2"

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
the existing `K_A` canonical. The parent table must be rewritten with certified
stable parent IDs and `K_A` rebuilt in a new versioned directory before the
single-step gate can pass. The readiness audit also verifies that every local
stable node exists in the registry and that its parent definition matches the
rebuilt table.

Do not resolve conflicting lineages by retaining the first row. Review them and
record the selected parent identities and provenance explicitly.

## Construction gate

Single-step `H` remains prohibited until all blocking reasons are cleared:

- source children have one selected lineage or a recorded conservative-founder resolution;
- children and parents use certified stable IDs;
- no self-parent or cyclic relationships exist;
- repeated parent IDs are explicitly reviewed as legitimate selfing records;
- the pedigree-node universe exactly matches the `K_A` order;
- both relationship kernels pass numerical integrity checks;
- `A22` and HMP `G` have sufficient overlap and valid scale alignment;
- the sampled blended genomic block is PSD.

Warnings about shallow lineage, weak relationship concordance, or poor
conditioning must be reported with any subsequent single-step experiment.
