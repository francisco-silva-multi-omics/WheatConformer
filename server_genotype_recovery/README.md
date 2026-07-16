# Recovered Genotype Panel Kernels

This workflow exhaustively inventories genotype-linked identifiers under
`GENOTYPIC_DATA`, resolves them against the canonical trial catalog, and builds
separate platform-specific kernels for raw panels missed by the preview audit.

It does not concatenate raw markers across platforms. Marker identity, allele
coding, missingness, ascertainment, and platform effects differ between 80k,
Seeds of Discovery DArTseq, and IWYP 35k. Each panel is parsed and certified
independently, then exposed to the multi-trait model as an opt-in partial expert.

## Supported Matrices

| Platform | Matrix orientation | Trial-ID resolution |
| --- | --- | --- |
| `80k_hexaploid` | sample by marker Flapjack text | canonical aliases and explicit IDs |
| `seeds_dartseq` | marker by sample text | `SampleIDvsGID_45610samples.txt` |
| `iwyp35k` | marker by sample with IWYP preamble | GID preamble |

The `.flapjack` files in the Seeds and Mexican datasets are SQLite project
containers. The genotype builder intentionally reads their text matrix mirrors;
the exhaustive audit records the containers in its file inventory.

## Server Execution

```bash
cd /DATA2/estancias/tesis_javier/model_DATA/genotipoXambiente
export PYTHON="$HOME/tools/tf_wheat_cpu/bin/python"

nohup bash scripts/run_genotypic_panel_recovery.sh . \
  > logs/genotypic_panel_recovery.nohup.log 2>&1 &
```

Run or retry selected panels without repeating the others:

```bash
PLATFORMS="80k_hexaploid" \
  nohup bash scripts/run_genotypic_panel_recovery.sh . \
  > logs/genotypic_panel_recovery_80k.nohup.log 2>&1 &

PLATFORMS="seeds_dartseq iwyp35k" SAVE_DOSAGE=0 \
  nohup bash scripts/run_genotypic_panel_recovery.sh . \
  > logs/genotypic_panel_recovery_seeds_iwyp.nohup.log 2>&1 &
```

The full run writes exhaustive match evidence, per-platform sample and marker
QC, linear VanRaden and Gaussian/RBF kernels, retained marker/sample orders,
kernel certification, and
`genotype_panels/recovered/recovered_genotype_kernel_manifest.tsv`.

Recovered kernels are `enabled_default=False`. They must pass ledger alignment
certification and multi-seed validation ablation before being admitted to the
quantitative baseline.

## Default QC

- sample missingness at most `0.20`;
- sample heterozygosity at most `0.20`;
- marker missingness at most `0.20`;
- marker heterozygosity at most `0.20`;
- minor allele frequency at least `0.01`;
- duplicate canonical GIDs resolved by lowest missingness, then lowest
  heterozygosity, then stable source-sample order;
- missing calls mean-imputed only after QC for VanRaden construction;
- linear kernel mean-diagonal scaled;
- RBF gamma recorded from the median positive pairwise distance heuristic.

The RBF kernel is generated for ablation but is not enabled automatically.
