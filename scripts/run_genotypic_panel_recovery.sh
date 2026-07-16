#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
PYTHON="${PYTHON:-python}"
PLATFORMS="${PLATFORMS:-80k_hexaploid seeds_dartseq iwyp35k}"
SAVE_DOSAGE="${SAVE_DOSAGE:-1}"
cd "$ROOT"

mkdir -p logs audit/genotypic_recovery genotype_panels/recovered

echo "[$(date '+%F %T')] START exhaustive GID recovery"
"$PYTHON" -m audit.recover_genotypic_gid_matches \
  --root . \
  --out-dir audit/genotypic_recovery
echo "[$(date '+%F %T')] DONE exhaustive GID recovery"

for platform in $PLATFORMS; do
  echo "[$(date '+%F %T')] START $platform kernel"
  args=(--root . --platform "$platform")
  if [[ "$SAVE_DOSAGE" == "1" ]]; then
    args+=(--save-dosage)
  fi
  "$PYTHON" -m server_genotype_recovery.build_platform_kernel "${args[@]}"
  echo "[$(date '+%F %T')] DONE $platform kernel"
done

"$PYTHON" -m server_genotype_recovery.combine_registry_fragments --root .

echo "[$(date '+%F %T')] Recovery outputs"
echo "  audit/genotypic_recovery/matrix_backed_gid_dataset_summary.tsv"
echo "  audit/genotypic_recovery/matrix_backed_gid_union_summary.tsv"
echo "  genotype_panels/recovered/recovered_genotype_kernel_manifest.tsv"
