#!/usr/bin/env bash
#
# Submit from the repository root:
#   sbatch scripts/run_multipref_downstream_array.slurm
#
# For a different number of splits, override both the array range and N_SPLITS:
#   sbatch --array=0-199 --export=ALL,N_SPLITS=200 scripts/run_multipref_downstream_array.slurm

#SBATCH --job-name=multipref_downstream
#SBATCH --array=0-99
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

N_SPLITS="${N_SPLITS:-100}"
OUTDIR="${OUTDIR:-src/results/multipref_downstream}"
FLAT="${FLAT:-src/results/multipref_v4/multipref_flat_annotations.csv}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" src/run_multipref_downstream_alignment.py \
  --flat "${FLAT}" \
  --n-splits "${N_SPLITS}" \
  --split-index "${SLURM_ARRAY_TASK_ID}" \
  --outdir "${OUTDIR}"
