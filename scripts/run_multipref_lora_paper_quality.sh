#!/usr/bin/env bash
#
# LoRA LLM reward-model experiment for the MultiPref cyclic-residual paper.
#
# Submit/run from the repository root on the GPU devbox:
#
#   N_SPLITS=10 EPOCHS=1 MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct GPUS="0 1" \
#     bash scripts/run_multipref_lora_paper_quality.sh
#
# This launches one split per GPU at a time, combines split outputs, and runs
# the same conservative paper-quality analysis used by the encoder RM checks.

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
MODEL_SLUG="$(printf "%s" "${MODEL_NAME}" | tr '/:.' '___' | tr -cs '[:alnum:]_' '_')"
OUTDIR="${OUTDIR:-src/results/multipref_lora_reward_${MODEL_SLUG}_full}"
N_SPLITS="${N_SPLITS:-10}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-4}"
MAX_LENGTH="${MAX_LENGTH:-768}"
PRECISION="${PRECISION:-fp32}"
NUM_WORKERS="${NUM_WORKERS:-2}"
GPUS_STRING="${GPUS:-0 1}"
PREF_GROUPS="${PREF_GROUPS:-all}"
ASPECTS="${ASPECTS:-overall helpful truthful harmless}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
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
echo "LoRA: r=${LORA_R} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT}"

pids=()
split=0
while [[ "${split}" -lt "${N_SPLITS}" ]]; do
  gpu="${GPU_LIST[$((split % ${#GPU_LIST[@]}))]}"
  log_path="${OUTDIR}/logs/split_${split}.log"
  echo "Launching split ${split} on GPU ${gpu}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" src/run_multipref_lora_reward_downstream.py \
    --model-name "${MODEL_NAME}" \
    --groups ${PREF_GROUPS} \
    --aspects ${ASPECTS} \
    --n-splits "${N_SPLITS}" \
    --split-index "${split}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --lr "${LR}" \
    --max-length "${MAX_LENGTH}" \
    --precision "${PRECISION}" \
    --num-workers "${NUM_WORKERS}" \
    --lora-r "${LORA_R}" \
    --lora-alpha "${LORA_ALPHA}" \
    --lora-dropout "${LORA_DROPOUT}" \
    --gradient-checkpointing \
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
"${PYTHON_BIN}" src/run_multipref_lora_reward_downstream.py \
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
