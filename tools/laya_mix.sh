#!/bin/bash
# Laya: mixed camp (r0-r3 aggregate) + lookahead-improved labels, then one DAgger round where Laya plays
# (GPU, recorded) and CPU workers replay + label its states with the improved teacher.
set -euo pipefail
L=/tmp/23ucs715_laya; D=$L/data; cd ~/zombiee-v3
export HF_HOME=$L/hf PYTHONUNBUFFERED=1 LAYA_CUDA_AMP=fp16 TRANSFORMERS_VERBOSITY=error
PY=$L/env/bin/python; CPY=~/miniconda3/envs/zombiee/bin/python
LOG=research_log/data/2026-10-09_laya_dagger.jsonl
pick() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<16000 && !f {print $1; f=1}' || true; }
gpu() { while true; do G1=$(pick); sleep 30; G2=$(pick); [ -n "$G1" ] && [ "$G1" = "$G2" ] && break; done; export CUDA_VISIBLE_DEVICES=$G1; echo "GPU $G1 $(date)"; }
mix() {  # out in1 in2 ...: strip extra keys, keep all non-wait rows, waits capped at half the non-wait count (per input)
$PY - "$@" <<'PYEOF'
import json, random, sys, collections
out, ins = sys.argv[1], sys.argv[2:]; allrows = []
for i, path in enumerate(ins):
    rows = [json.loads(l) for l in open(path)]
    rows = [{"state": r["state"], "questions": r["questions"], "expected": r["expected"]} for r in rows]
    w = [r for r in rows if r["expected"]["action"] == "wait"]; nw = [r for r in rows if r["expected"]["action"] != "wait"]
    random.Random(i).shuffle(w); allrows += nw + w[:len(nw) // 2]
    print("  ", path, len(rows), "->", len(nw) + min(len(w), len(nw) // 2))
random.Random(99).shuffle(allrows)
with open(out, "w") as f:
    for r in allrows: f.write(json.dumps(r) + "\n")
print("mix:", len(allrows), dict(collections.Counter(r["expected"]["action"] for r in allrows).most_common()))
PYEOF
}
ft() {  # data init out epochs
  for TRY in 1 2 3 4 5; do
    gpu; rm -rf "$3"
    if $PY - "$1" "$2" "$3" "$4" <<'PYEOF'
import json, sys
from laya.train import TrainConfig, finetune
cfg = TrainConfig(epochs=int(sys.argv[4]), micro_batch=4, grad_accum=8, encoder_lr=1.5e-5, head_lr=6e-5, log_every=400)
print(json.dumps(finetune(sys.argv[1], sys.argv[2], sys.argv[3], cfg), default=str)[:600])
PYEOF
    then return 0; fi
    echo "fine-tune attempt $TRY failed $(date)"
  done
  return 1
}
run() { for TRY in 1 2 3; do gpu; if "$@"; then return 0; fi; echo "attempt $TRY failed: $* $(date)"; done; return 1; }

echo "=== A: mixed fine-tune $(date)"
mix $D/train_mix.jsonl $D/agg.jsonl $D/la_labels.jsonl
ft $D/train_mix.jsonl $L/ckpt/laya_r3 $L/ckpt/laya_mix 1
run $PY -m training.laya_policy eval --student $L/ckpt/laya_mix --n-eval 200 --log $LOG

echo "=== B: Laya plays 150 recorded episodes $(date)"
run $PY -m training.laya_policy record --student $L/ckpt/laya_mix --episodes 150 --seed 6161 --out $D/rec_mix.jsonl

echo "=== C: CPU replay + improved-teacher labels $(date)"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 $CPY tools/probes/lookahead_labels.py 150 $D/dag_la.jsonl --replay $D/rec_mix.jsonl --m 16 --start 55 --delta 0.05 --workers 64

echo "=== D: fine-tune on mix + DAgger rows $(date)"
mix $D/train_mix2.jsonl $D/agg.jsonl $D/la_labels.jsonl $D/dag_la.jsonl
ft $D/train_mix2.jsonl $L/ckpt/laya_mix $L/ckpt/laya_mix2 1
run $PY -m training.laya_policy eval --student $L/ckpt/laya_mix2 --n-eval 200 --log $LOG
echo "=== LAYA MIX DONE $(date)"
