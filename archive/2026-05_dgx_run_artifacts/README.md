# DGX run artifacts — recovered 2026-09-08

This folder is the **raw evidence behind the results table and postmortem in the
top-level README**. It was sitting untracked on the LNMDGX DGX box
(`/home/23ucs715/archive/hackathon/zombiee-v2/`) and was never committed — the
`.gitignore` deliberately excludes `eval_results/*.json`, `eval_results/*.png`,
and `*.log`. That box is being wiped for disk space, so the small, high-value
files are committed here before deletion.

## What was happening on that machine (not in any md file)

The README's "4 runs across 90 hours" table is a summary. Here is the blow-by-blow
that the logs and eval JSON actually show:

| run | dates | what was tried | checkpoints eval'd | outcome (from the JSON here) |
|---|---|---|---|---|
| **1** | 2026-04-27 → 04-28 | first single-action GRPO, 100-step episodes, 15-component rubric | 13, 22, 24, 27, 30 | trained mean reward **0.87** vs baseline **1.06** — *worse than untrained*. `scan` action ~49%. 0% survival. |
| **1b** | 2026-04-28 | re-eval of the same checkpoints with `trained_is_real=false` fallback path | 22, 24, 27, 30 | eval harness couldn't load the adapter and silently scored the **base model** (reward 1.063 flat). This is why several JSON files show identical trained/baseline numbers — they are not real trained evals. |
| **2** | 2026-04-30 | scan-streak rubric | aborted @ step 21 | scan-streak threshold of 2 is unreachable when each rollout is 1 action — logged, abandoned. |
| **3** | 2026-05-09 → 05-10 | flat per-scan penalty + forage-shaping at hunger≥1 (`4f34535`, `95b3d57`) | 25 (`ckpt-25-run3`) | scan 49% → 41%, trained reward **0.719** vs baseline 0.891. Still 0% survival. |
| **4** | 2026-05-10 | **multi-action GRPO** `--prefix-actions 5` (`09a4717`), + inner-ring food/water (`741f4b6`), + heuristic forage-threshold fix (`33e1df9`) | 25, 40, 60 (`*-run4-best`) | first win: trained reward **1.109–1.128** vs baseline **1.028**. `pct_reached_step_30` still **0.0**. Survival still 0%. |
| **5 / GiGPO** | 2026-05-11 onward | pivot from GRPO to GiGPO step-level credit assignment | — | committed to branch `claude/phase-2-gigpo` (`6b2ae2b`…`628b093`, through 2026-07-01), not run on this box. |

### The smoking gun the README mentions: KL = 0

`logs/train100_20260510_131411.log` (run 4, 60 steps, 5 h 27 min on one V100):

```
[metrics] step 60: {'loss': 0.0604, 'grad_norm': 24.78,
  'learning_rate': 1.67e-07, 'reward': -0.757, 'reward_std': 0.196,
  'completion_length': 96.75, 'kl': 0.604 }
```

…and run 3 (`train100_20260510_021813.log`, 100 steps, 4 h 21 min) ended at
`reward=-0.185, kl=0.473, completion_length=29.5` with the action histogram
collapsed to `actions[drink=6, scan=2]` — i.e. mode collapse. Training reward is
negative throughout because the rubric's dense shaping terms dominate; the eval
reward (0.7–1.1) is a different scale.

### Eval numbers, extracted from the JSON in this folder

| eval file | ckpt | `trained_is_real` | baseline mean reward | trained mean reward | survival | `pct_reached_step_30` |
|---|---|---|---|---|---|---|
| `eval_results/eval_step_0013.json` | 13 | true | 1.06 | 0.87 | 0% | — |
| `eval_results/ckpt-25-run3/…0025.json` | 25 | true | 0.891 | 0.719 | 0% | — |
| `eval_results/ckpt-25-run4/…0025.json` | 25 | true | 0.891 | 0.707 | 0% | — |
| `eval_results/ckpt-25-run4-matched/…` | 25 | true | 0.891 | 0.895 | 0% | — |
| `eval_results/ckpt-25-run4-fixed-env/…` | 25 | true | 1.028 | **1.124** | 0% | 0.0 |
| `eval_results/ckpt-40-run4-best/…0040.json` | 40 | true | 1.028 | **1.109** | 0% | 0.0 |
| `eval_results/ckpt-60-run4-best/…0060.json` | 60 | true | 1.028 | **1.128** | 0% | 0.0 |
| `eval_results/ckpt-100/…0100.json` | 100 | true | 1.064 | 0.837 | 0% | — |
| `eval_step_0022/0024/0027/0030.json` (root) | 22–30 | **false** | 1.06 | 1.063 | 0% | — (base-model fallback, ignore) |

Bottom line the README already states honestly: **survival rate never left 0%.**
The run-4 multi-action change is the only configuration where the trained policy
beat the untrained baseline on mean reward, and only by ~0.1.

## Contents

- `eval_results/` — every eval JSON + the `*_bars.png` / `eval_history.png` charts,
  for runs 1, 3, and 4 (incl. the `run4-best`, `run4-matched`, `run4-fixed-env`
  variants). `per_episode` arrays are included in the JSON.
- `logs/` — 34 logs: `train100_*.log` (4 training runs) and `eval_ckpt*_*.log`
  (per-checkpoint eval console output, with per-step `[v2 STEP]` traces).
- `git-log-at-recovery.txt` — full `git log --all` as seen on the box on
  2026-09-08, including the `origin/claude/phase-2-gigpo` commits up to 2026-07-01.

## What was NOT recovered (deleted with the box)

- All LoRA checkpoint weights: `checkpoints_run2/`, `checkpoints_run3/`,
  `checkpoints_30step_apr28/`, `checkpoints_old_scanspam_20260509/` — ~13.4 GB of
  `adapter_model.safetensors` + optimizer state across ~30 checkpoint steps.
  Reproducible from `training/train.py` + the Phase-1 hyperparameters in commits
  `318f094`…`0193abb` on a fresh V100/A10G run.
- `checkpoints/metrics.jsonl` and `train_summary.json` (gitignored, not copied
  in time — the numbers above come from the training-log `[metrics]` lines).
