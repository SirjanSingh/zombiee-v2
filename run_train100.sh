#!/usr/bin/env bash
# Train for 50 steps on a free V100 (GPU 4 or 7) then generate plots.
# V100 is sm_70 — no bf16, no fused-adam, but 32 GB so --no-4bit is safe.
# Picks GPU 4 by default (32 GB completely free); change CUDA_VISIBLE_DEVICES
# to 7 if someone else claims 4 before you start.

set -euo pipefail

CUDA_VISIBLE_DEVICES=4          # GPU 4: 32 GB nearly free
STEPS=100
OUTDIR="./checkpoints"
METRICS="${OUTDIR}/metrics.jsonl"
PLOTS_DIR="./plots"
LOG_DIR="./logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train100_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR" "$PLOTS_DIR"

echo "[run_train100] $(date): starting ${STEPS}-step run on GPU ${CUDA_VISIBLE_DEVICES}"
echo "[run_train100] log -> ${LOG_FILE}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python -m training.train \
    --max-steps          ${STEPS} \
    --save-steps         1 \
    --save-total-limit   ${STEPS} \
    --num-generations    12 \
    --max-completion-length 512 \
    --rollout-limit      60 \
    --no-4bit \
    --optim              adamw_torch \
    --lr                 1e-5 \
    --grad-accum-steps   4 \
    --output-dir         "${OUTDIR}" \
    2>&1 | tee "${LOG_FILE}"

echo "[run_train100] $(date): training done — generating plots"

python -m training.plots \
    --metrics-file       "${METRICS}" \
    --eval-results-dir   ./eval_results \
    --output-dir         "${PLOTS_DIR}" \
    2>&1 | tee -a "${LOG_FILE}"

echo "[run_train100] $(date): done. Plots in ${PLOTS_DIR}/"
