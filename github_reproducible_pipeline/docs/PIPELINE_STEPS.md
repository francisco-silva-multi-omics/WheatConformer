# Pipeline Steps

## Phase 0: Prepare Workspace

```bash
python scripts/00_prepare_workspace_from_raw.py \
  --raw-dir "$RAW_DATA_DIR" \
  --work-dir "$WORK_DIR"
```

## Phase 1: Core Preprocessing

```bash
bash scripts/01_run_core_pipeline.sh
```

This builds:

```text
metadata_outputs/
genotype_panels/hmp/
genotype_panels/mas/
genotype_panels/dartag/
genotype_panels/dartseq_landrace/
genotype_panels/gbs_sawyt/
phenotypes/
environment/
functional_annotation/
integrated_database/
```

## Phase 2: 80k Full Server Priors

```bash
sbatch server_80k_pipeline/run_80k_pipeline.slurm
```

## Phase 3: Stage-1 Model Inputs

```bash
bash scripts/02_run_model_inputs.sh
```

## Phase 4: TensorFlow Baseline Training

```bash
bash scripts/03_run_training.sh
```

or submit the individual SLURM files:

```bash
sbatch server_training_pipeline/run_multikernel_training.slurm
sbatch server_gbs_pipeline/run_gbs_multikernel_training.slurm
```

## Phase 5: Regulatory TensorFlow Model

```bash
bash scripts/04_run_regulatory_enformer_tf.sh
```

or on SLURM:

```bash
sbatch server_training_pipeline/run_enformer_like_training.slurm
```

## Check Outputs

```bash
python scripts/check_expected_outputs.py
```
