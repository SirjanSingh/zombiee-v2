#!/bin/bash
# Laya continued fine-tune on lookahead-improved labels (tools/probes/lookahead_labels.py), from laya_r3.
set -euo pipefail
L=/tmp/23ucs715_laya; cd ~/zombiee-v3
export HF_HOME=$L/hf PYTHONUNBUFFERED=1 LAYA_CUDA_AMP=fp16 TRANSFORMERS_VERBOSITY=error
PY=$L/env/bin/python
LOG=research_log/data/2026-10-09_laya_dagger.jsonl
while kill -0 $(cat ~/lal.pid) 2>/dev/null; do sleep 60; done          # wait for the label job
grep -q "^done:" logs/la_labels.log || { echo "label job did not finish cleanly"; exit 1; }
pick() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<16000 && !f {print $1; f=1}' || true; }
gpu() { while true; do G1=$(pick); sleep 30; G2=$(pick); [ -n "$G1" ] && [ "$G1" = "$G2" ] && break; done; export CUDA_VISIBLE_DEVICES=$G1; echo "GPU $G1 $(date)"; }
$PY - <<'PYEOF'
import json, random, collections
rows = [json.loads(l) for l in open("/tmp/23ucs715_laya/data/la_labels.jsonl")]
keep = [{"state": r["state"], "questions": r["questions"], "expected": r["expected"]} for r in rows]
w = [r for r in keep if r["expected"]["action"] == "wait"]; nw = [r for r in keep if r["expected"]["action"] != "wait"]
random.Random(0).shuffle(w); out = nw + w[:len(nw) // 2]; random.Random(1).shuffle(out)
with open("/tmp/23ucs715_laya/data/train_la.jsonl", "w") as f:
    for r in out: f.write(json.dumps(r) + "\n")
print("la labels:", len(rows), "overrides", sum(bool(r.get("override")) for r in rows), "-> train", len(out),
      dict(collections.Counter(r["expected"]["action"] for r in out).most_common()))
PYEOF
for TRY in 1 2 3 4 5; do
  gpu; rm -rf $L/ckpt/laya_la
  if $PY - <<'PYEOF'
import json
from laya.train import TrainConfig, finetune
cfg = TrainConfig(epochs=2, micro_batch=4, grad_accum=8, encoder_lr=1.5e-5, head_lr=6e-5, log_every=200)
print(json.dumps(finetune("/tmp/23ucs715_laya/data/train_la.jsonl", "/tmp/23ucs715_laya/ckpt/laya_r3",
                          "/tmp/23ucs715_laya/ckpt/laya_la", cfg), default=str)[:800])
PYEOF
  then break; fi
  echo "fine-tune attempt $TRY failed $(date)"
done
$PY -m training.laya_policy eval --student $L/ckpt/laya_la --n-eval 200 --log $LOG
echo "=== LAYA LA DONE $(date)"
