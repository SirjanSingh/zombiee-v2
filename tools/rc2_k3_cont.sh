#!/bin/bash
# v3-rc2 DAgger K=3, rounds 4-5, continuing from sft_rc2_k3_r3 (A0 lifetime 80, extracted 0%).
# Why: round R labels states reached by student R-1; r3 was trained on states from r2 (died ~t27),
# so the late game (eat ~t67, water refill, helicopter t90) was never labelled. 85% of the
# aggregate is t<30, so each round trains on all t>=30 rows + a 2500-row sample of t<30 rows.
set -euo pipefail
cd ~/zombiee-v3
pick() { nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '$2<6000 && $3<60 && !f {print $1; f=1}' || true; }
echo "waiting for a free GPU $(date)"
while true; do G1=$(pick); sleep 60; G2=$(pick); [ -n "$G1" ] && [ "$G1" = "$G2" ] && break; done
PY=~/miniconda3/envs/zombiee/bin/python
export CUDA_VISIBLE_DEVICES=$G1 PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=error SC_STEP_LOG_EVERY=0
B=v3-rc2; K=3
echo "=== rc2 K=3 rounds 4-5 on GPU $G1 $(date)"
STUDENT=checkpoints/sft_rc2_k3_r3
for R in 4 5; do
  echo "=== round $R $(date)"
  $PY -m training.eval_v3 --balance $B --prefix-actions $K --lora-path $STUDENT --n-episodes 300 --temperature 0.7 \
      --seed $((8000+R)) --baselines --no-log --dagger-out data/dagger_rc2_k3_r$R.jsonl
  cat data/dagger_rc2_k3_r$R.jsonl >> data/dagger_rc2_k3_agg.jsonl
  $PY - "$R" <<'EOF'
import json, random, sys, collections
R = sys.argv[1]
rows = [json.loads(l) for l in open("data/dagger_rc2_k3_agg.jsonl")]
late = [r for r in rows if (r.get("t") or 0) >= 30]
early = [r for r in rows if (r.get("t") or 0) < 30]
random.Random(int(R)).shuffle(early)
keep = late + early[:2500]
with open(f"data/dagger_rc2_k3_train_r{R}.jsonl", "w") as f:
    for r in keep:
        f.write(json.dumps(r) + "\n")
b = collections.Counter((r.get("t") or 0) // 15 * 15 for r in keep)
print(f"round {R}: agg {len(rows)} -> train {len(keep)} (late {len(late)}, early kept {min(2500, len(early))})",
      sorted(b.items()))
EOF
  $PY -m training.sft --data data/dagger_rc2_k3_train_r$R.jsonl --output-dir checkpoints/sft_rc2_k3_r$R
  STUDENT=checkpoints/sft_rc2_k3_r$R
  $PY -m training.eval_v3 --balance $B --prefix-actions $K --lora-path $STUDENT --n-episodes 30 --baselines --tag rc2-k3-r$R
done
$PY -m training.eval_v3 --balance $B --prefix-actions $K --lora-path $STUDENT --n-episodes 60 --seed 4321 \
    --baselines camp heuristic_v3 --record-replays 4 --tag rc2-k3-r5-final
echo "=== RC2 K3 CONT DONE $(date)"
