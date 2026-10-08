#!/bin/bash
# Resume v3-rc2 DAgger K=3 round 5 at the SFT step (r5 data already collected; first SFT OOM'd on a shared GPU 0).
set -euo pipefail
cd ~/zombiee-v3
pick() { nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '$2<6000 && $3<60 && !f {print $1; f=1}' || true; }
echo "waiting for a free GPU $(date)"
while true; do G1=$(pick); sleep 60; G2=$(pick); [ -n "$G1" ] && [ "$G1" = "$G2" ] && break; done
PY=~/miniconda3/envs/zombiee/bin/python
export CUDA_VISIBLE_DEVICES=$G1 PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=error SC_STEP_LOG_EVERY=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B=v3-rc2; K=3
echo "=== round 5 SFT on GPU $G1 $(date)"
rm -rf checkpoints/sft_rc2_k3_r5
$PY -m training.sft --data data/dagger_rc2_k3_train_r5.jsonl --output-dir checkpoints/sft_rc2_k3_r5
STUDENT=checkpoints/sft_rc2_k3_r5
$PY -m training.eval_v3 --balance $B --prefix-actions $K --lora-path $STUDENT --n-episodes 30 --baselines --tag rc2-k3-r5
$PY -m training.eval_v3 --balance $B --prefix-actions $K --lora-path $STUDENT --n-episodes 60 --seed 4321 \
    --baselines camp heuristic_v3 --record-replays 4 --tag rc2-k3-r5-final
echo "=== RC2 K3 R5 DONE $(date)"
