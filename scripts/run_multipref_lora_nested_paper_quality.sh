#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
N_SPLITS="${N_SPLITS:-5}"
EPOCHS="${EPOCHS:-1}"
GPUS_STRING="${GPUS:-0 1}"
GROUPS_STRING="${GROUPS:-all}"
ASPECTS_STRING="${ASPECTS:-overall helpful truthful harmless}"
OUTDIR="${OUTDIR:-src/results/multipref_lora_nested_Qwen_Qwen2_5_1_5B_Instruct_full}"
PRECISION="${PRECISION:-fp32}"
MAX_LENGTH="${MAX_LENGTH:-768}"
INTERACTION_RANK="${INTERACTION_RANK:-16}"
N_BOOTSTRAP="${N_BOOTSTRAP:-5000}"

read -r -a GPU_IDS <<< "${GPUS_STRING}"
read -r -a GROUP_NAMES <<< "${GROUPS_STRING}"
read -r -a ASPECT_NAMES <<< "${ASPECTS_STRING}"
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "GPUS must contain at least one CUDA device." >&2
  exit 2
fi

mkdir -p "${OUTDIR}/logs"
declare -a SLOT_PIDS

echo "Model: ${MODEL_NAME}"
echo "Output: ${OUTDIR}"
echo "Prompt folds: ${N_SPLITS}"
echo "GPUs: ${GPUS_STRING}"
echo "Interaction rank: ${INTERACTION_RANK}"

for ((split_index = 0; split_index < N_SPLITS; split_index++)); do
  slot=$((split_index % ${#GPU_IDS[@]}))
  gpu="${GPU_IDS[$slot]}"
  if [[ -n "${SLOT_PIDS[$slot]:-}" ]]; then
    wait "${SLOT_PIDS[$slot]}"
  fi
  log_path="${OUTDIR}/logs/split_${split_index}.log"
  echo "Launching fold ${split_index} on GPU ${gpu}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" python src/run_multipref_lora_nested_reward_downstream.py \
    --model-name "${MODEL_NAME}" \
    --groups "${GROUP_NAMES[@]}" \
    --aspects "${ASPECT_NAMES[@]}" \
    --n-splits "${N_SPLITS}" \
    --split-index "${split_index}" \
    --epochs "${EPOCHS}" \
    --batch-size 2 \
    --eval-batch-size 4 \
    --grad-accum 8 \
    --lr 1e-4 \
    --max-length "${MAX_LENGTH}" \
    --precision "${PRECISION}" \
    --num-workers 2 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --interaction-rank "${INTERACTION_RANK}" \
    --interaction-dropout 0.1 \
    --gradient-checkpointing \
    --tie-training half \
    --include-same-model-train \
    --outdir "${OUTDIR}" \
    >"${log_path}" 2>&1 &
  SLOT_PIDS[$slot]=$!
done

for pid in "${SLOT_PIDS[@]}"; do
  wait "${pid}"
done

echo "Combining fold-level region outputs."
python src/run_multipref_lora_nested_reward_downstream.py \
  --combine-glob "${OUTDIR}/*_split_*_region_results.csv" \
  --outdir "${OUTDIR}"

echo "Running matched-model inference."
python src/analyze_multipref_nested_reward_results.py \
  --region-glob "${OUTDIR}/*_split_*_region_results.csv" \
  --prediction-glob "${OUTDIR}/*_split_*_predictions.csv" \
  --outdir "${OUTDIR}/paper_quality" \
  --n-bootstrap "${N_BOOTSTRAP}"

echo "Done. Report: ${OUTDIR}/paper_quality/nested_reward_report.md"
