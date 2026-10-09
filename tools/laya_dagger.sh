#!/bin/bash
# Laya (421M) DAgger on v3-rc2 vs the camp teacher. Lives in /tmp (outside the 90 GB home quota).
# round 0 = behaviour cloning on camp rollouts; rounds 1-3 = Laya plays, camp labels; every round
# fine-tunes laya-typed-decisions from scratch on the aggregate (waits subsampled to 1/3 of rows).
set -euo pipefail
L=/tmp/23ucs715_laya; cd ~/zombiee-v3
export HF_HOME=$L/hf PYTHONUNBUFFERED=1 LAYA_CUDA_AMP=fp16 TRANSFORMERS_VERBOSITY=error
PY=$L/env/bin/python
BASE=$(ls -d $L/hf/hub/models--convaiinnovations--laya-typed-decisions/snapshots/*)
pick() { nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '$2<16000 && !f {print $1; f=1}' || true; }
gpu() { while true; do G1=$(pick); sleep 30; G2=$(pick); [ -n "$G1" ] && [ "$G1" = "$G2" ] && break; done; export CUDA_VISIBLE_DEVICES=$G1; echo "GPU $G1 $(date)"; }
LOG=research_log/data/2026-10-09_laya_dagger.jsonl

balance() {  # $1 = aggregate jsonl, $2 = out; keep all non-wait rows, waits capped at half the non-wait count
$PY - "$1" "$2" <<'PYEOF'
import json, random, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
w = [r for r in rows if r["expected"]["action"] == "wait"]; nw = [r for r in rows if r["expected"]["action"] != "wait"]
random.Random(0).shuffle(w); keep = nw + w[:len(nw) // 2]; random.Random(1).shuffle(keep)
with open(sys.argv[2], "w") as f:
    for r in keep: f.write(json.dumps(r) + "\n")
print("balance:", len(rows), "->", len(keep), dict(collections.Counter(r["expected"]["action"] for r in keep).most_common()))
PYEOF
}
ft_once() {  # $1 = data, $2 = out
$PY - "$1" "$2" "$BASE" <<'PYEOF'
import sys, json
from laya.train import TrainConfig, finetune
cfg = TrainConfig(epochs=3, micro_batch=4, grad_accum=8, log_every=200)
print(json.dumps(finetune(sys.argv[1], sys.argv[3], sys.argv[2], cfg), default=str)[:1500])
PYEOF
}

ft() {  # retry on OOM (GPUs are shared and other jobs grow): wait for a GPU again each time
  for TRY in 1 2 3 4 5; do
    rm -rf "$2"
    if ft_once "$1" "$2"; then return 0; fi
    echo "fine-tune attempt $TRY failed $(date); waiting for a GPU"; gpu
  done
  return 1
}

echo "=== laya DAgger start $(date)"
if [ ! -f $L/data/r0.jsonl ]; then          # resume: zero-shot eval + round-0 rows already done
  gpu
  $PY -m training.laya_policy eval --student $BASE --n-eval 30 --log $LOG          # zero-shot baseline
  $PY -m training.laya_policy rows --teacher camp --episodes 300 --seed 5150 --out $L/data/r0.jsonl
  cp $L/data/r0.jsonl $L/data/agg.jsonl
fi
STUDENT=""
for R in 0 1 2 3; do
  echo "=== round $R $(date)"
  if [ "$R" -gt 0 ]; then
    gpu
    $PY -m training.laya_policy rows --teacher camp --student $STUDENT --episodes 150 --seed $((5150+R)) --out $L/data/r$R.jsonl
    cat $L/data/r$R.jsonl >> $L/data/agg.jsonl
  fi
  balance $L/data/agg.jsonl $L/data/train_r$R.jsonl
  gpu
  ft $L/data/train_r$R.jsonl $L/ckpt/laya_r$R
  $PY -m training.laya_policy eval --student $L/ckpt/laya_r$R --n-eval 200 --log $LOG
  [ -n "$STUDENT" ] && [ "$STUDENT" != "$L/ckpt/laya_r$R" ] && rm -rf "$STUDENT"      # keep latest only
  STUDENT=$L/ckpt/laya_r$R
done
echo "=== LAYA DONE $(date)"
