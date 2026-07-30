#!/usr/bin/env bash
#
# Full neural reward-model experiment for the MultiPref cyclic-residual paper.
#
# Submit/run from the repository root on the GPU devbox:
#
#   bash scripts/run_multipref_neural_paper_quality.sh
#
# Useful overrides:
#
#   N_SPLITS=10 EPOCHS=2 MODEL_NAME=roberta-large \
#     bash scripts/run_multipref_neural_paper_quality.sh
#
#   N_SPLITS=10 EPOCHS=2 MODEL_NAME=microsoft/deberta-v3-large \
#     bash scripts/run_multipref_neural_paper_quality.sh
#
#   GPUS="0 1" BATCH_SIZE=8 PRECISION=bf16 \
#     bash scripts/run_multipref_neural_paper_quality.sh
#
# The script launches one split per GPU at a time, combines split outputs, and
# runs the conservative paper-quality analysis.

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-roberta-large}"
MODEL_SLUG="$(printf "%s" "${MODEL_NAME}" | tr '/:.' '___' | tr -cs '[:alnum:]_' '_')"
OUTDIR="${OUTDIR:-src/results/multipref_neural_reward_${MODEL_SLUG}_full}"
N_SPLITS="${N_SPLITS:-10}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-512}"
PRECISION="${PRECISION:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
GPUS_STRING="${GPUS:-0 1}"
PREF_GROUPS="${PREF_GROUPS:-all normal expert}"
ASPECTS="${ASPECTS:-overall helpful truthful harmless}"
N_BOOTSTRAP="${N_BOOTSTRAP:-2000}"
N_PERMUTATIONS="${N_PERMUTATIONS:-5000}"
PYTHON_BIN="${PYTHON_BIN:-python}"

read -r -a GPU_LIST <<< "${GPUS_STRING}"
if [[ "${#GPU_LIST[@]}" -lt 1 ]]; then
  echo "No GPUs specified. Set GPUS=\"0 1\" or similar." >&2
  exit 1
fi

mkdir -p "${OUTDIR}/logs"

echo "Model: ${MODEL_NAME}"
echo "Output: ${OUTDIR}"
echo "Splits: ${N_SPLITS}"
echo "GPUs: ${GPU_LIST[*]}"
echo "Preference groups: ${PREF_GROUPS}"
echo "Aspects: ${ASPECTS}"

pids=()
split=0
while [[ "${split}" -lt "${N_SPLITS}" ]]; do
  gpu="${GPU_LIST[$((split % ${#GPU_LIST[@]}))]}"
  log_path="${OUTDIR}/logs/split_${split}.log"
  echo "Launching split ${split} on GPU ${gpu}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" src/run_multipref_neural_reward_downstream.py \
    --model-name "${MODEL_NAME}" \
    --groups ${PREF_GROUPS} \
    --aspects ${ASPECTS} \
    --n-splits "${N_SPLITS}" \
    --split-index "${split}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --lr "${LR}" \
    --max-length "${MAX_LENGTH}" \
    --precision "${PRECISION}" \
    --num-workers "${NUM_WORKERS}" \
    --outdir "${OUTDIR}" \
    >"${log_path}" 2>&1 &
  pids+=("$!")

  if [[ "${#pids[@]}" -ge "${#GPU_LIST[@]}" ]]; then
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
    pids=()
  fi
  split=$((split + 1))
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "Combining split outputs."
"${PYTHON_BIN}" src/run_multipref_neural_reward_downstream.py \
  --combine-glob "${OUTDIR}/*_split_*_region_results.csv" \
  --outdir "${OUTDIR}"

echo "Running paper-quality inference."
"${PYTHON_BIN}" src/analyze_multipref_neural_paper_results.py \
  --region-glob "${OUTDIR}/*_split_*_region_results.csv" \
  --outdir "${OUTDIR}/paper_quality" \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --n-permutations "${N_PERMUTATIONS}" \
  --scope-summary

echo "Done. Main report:"
echo "${OUTDIR}/paper_quality/neural_reward_paper_report.md"
