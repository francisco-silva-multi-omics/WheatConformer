# Server Run From Naive Data

This is the intended reproducible server path when starting from the raw/naive data folders.

## 1. Clone And Create Environment

```bash
git clone <repo-url> wheat_gxe_pipeline
cd wheat_gxe_pipeline

conda create -n wheat80k -y python=3.11
conda activate wheat80k
python -m pip install -r requirements/base.txt
```

For TensorFlow training, use a GPU environment:

```bash
conda create -n wheattrain -y python=3.11
conda activate wheattrain
python -m pip install -r requirements/training_tensorflow.txt
```

## 2. Run CPU Processing From Raw Data

Use symlinks on the server unless you intentionally want a full copy.

```bash
conda activate wheat80k
export RAW_DATA_DIR=/DATA2/estancias/tesis_javier/raw_naive_data
export PREPARE_MODE=symlink
export FETCH_WEATHER=1
bash scripts/00_run_from_naive_data.sh
```

Equivalent SLURM submission:

```bash
sbatch --export=ALL,RAW_DATA_DIR=/DATA2/estancias/tesis_javier/raw_naive_data,PREPARE_MODE=symlink,FETCH_WEATHER=1 \
  scripts/run_from_naive_data.slurm
```

Outputs:

```text
metadata_outputs/
genotype_panels/
phenotypes/
environment/
integrated_database/
model_kernels/
logs/naive_data_validation.tsv
logs/expected_outputs.tsv
```

## 3. Run TensorFlow Training

Run after CPU processing completes.

```bash
conda activate wheattrain
bash scripts/03_run_training.sh
```

Or submit the training SLURM scripts:

```bash
sbatch server_training_pipeline/run_multikernel_training.slurm
sbatch server_gbs_pipeline/run_gbs_multikernel_training.slurm
```

## 4. Regulatory Model

Provide the IWGSC/Chinese Spring reference and multi-omics tracks:

```text
reference/IWGSC_RefSeq_v1.0.fa
reference/IWGSC_RefSeq_v1.0.fa.fai
multi_omics_data/*.bw
multi_omics_data/*.bed
```

Then run:

```bash
export REFERENCE_FASTA=/path/to/IWGSC_RefSeq_v1.0.fa
bash scripts/04_run_regulatory_enformer_tf.sh
```

## 5. Methodology Readiness Check

This reports which thesis-level model components exist and which remain missing.

```bash
python scripts/05_check_model_methodology_readiness.py
```

Expected remaining gaps until explicitly implemented:

```text
genotype_panels/pedigree/K_A.npy
model_kernels/K_z.npy
pangenome_resources/graph/marker_to_graph_interval.tsv
pangenome_resources/graph/genotype_path_dictionary.tsv
trained_models/validation_ablation_report.tsv
```
