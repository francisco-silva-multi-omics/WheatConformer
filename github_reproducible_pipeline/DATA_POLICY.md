# Data Policy

This Git repository should not contain raw or generated biological datasets.

Do not commit:

- raw CIMMYT fieldbooks, phenotype spreadsheets, or SQL dumps;
- 80k diversity panel genotype files;
- HMP, DArTseq, DArTAG, MAS, or 35K array genotype matrices;
- `.parquet`, `.npy`, `.bw`, `.bed`, `.vcf`, FASTA/GFF, graph pangenome, or multi-omics files;
- generated kernels, model-ready phenotype tables, or canonical integrated databases;
- files with access restrictions or redistribution terms.

Commit only:

- pipeline source code;
- lightweight documentation;
- environment/requirements files;
- small schema examples if they contain no restricted data.

Large reproducible outputs should be regenerated from the pipeline or stored in an approved institutional storage location, not GitHub.
