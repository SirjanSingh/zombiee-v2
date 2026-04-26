---
title: SurviveCity v2 — GRPO training
emoji: 🧟
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
suggested_hardware: a10g-large
suggested_storage: small
short_description: 5-agent zombie-survival GRPO training (Qwen 2.5-3B + LoRA)
---

<!--
HF Spaces ALWAYS uses `Dockerfile` at the repo root. There is no
`dockerfile:` front-matter field, so we keep the training container at
the root path:

  - `Dockerfile`             = training container. CMD launches GRPO +
                                a tiny FastAPI status sidecar on port
                                7860 so the Space framework's health
                                check passes. This is what HF builds.
  - `Dockerfile.envserver`   = env-only OpenEnv submission image (the
                                old root Dockerfile, preserved). Build
                                manually with
                                `docker build -f Dockerfile.envserver .`
                                if you need just the FastAPI env server
                                on port 7861.
  - `Dockerfile.dgx`         = byte-identical alias of `Dockerfile`,
                                kept so existing scripts/docs that
                                reference it still work.
  - `Dockerfile.dgx.bak`     = original DGX-validated 30 GB container
                                (April 2026), preserved for reproducibility.
-->

# SurviveCity v2 (v2.1 reward-fix revision)

5-agent multi-resource zombie survival env with **bite transmission**,
**spawn waves**, **iterated voting (3 rounds)**, **broadcast economy**,
**day/night cycle**, and **per-agent inventory**. Action space is a strict
superset of v1, so a v1 LoRA loads zero-shot.

The v2.1 revision (2026-04-26) fixes the four reward-function pathologies
that produced the floor-pinned, no-gradient training observed in v1 and
the v2.0 first pass:

1. **Per-agent damage credit assignment** — zombie damage to a non-acting
   agent and biter attacks during another agent's turn used to be wiped
   by `reset_step_flags()` before the victim's reward was computed. v2.1
   routes such damage through a per-agent `pending_reward` accumulator
   that the env drains into `cumulative_rewards` at end-of-round, so the
   credit-assignment trail is no longer broken.
2. **Forage / zombie-proximity / anti-camp shaping** — three new dense
   rubrics (`forage_shaping_reward`, `zombie_proximity_reward`,
   `anti_camp_reward`) plus a starting-infected `infected_deception_reward`
   give the policy a learnable gradient before the rare moment-of-eating
   bonus fires. Without these, eval transcripts showed agents wandering
   randomly until starvation around step 18 and never discovering food.
3. **Heuristic rollout in GRPO** — the v2.0 reward function applied the
   model's first action then rolled out 30 random actions. Random policy
   on a 15×15 grid almost never reaches food, so every group's cumulative
   reward floored at the same negative value (`reward_std ≈ 0` → no
   gradient). v2.1 uses the new `forage_heuristic_action` for the rollout
   so the cumulative reward distribution actually has variance.
4. **Terminal settlement for all agents** — `compose_reward` for the last
   actor used to fire BEFORE `advance_step` set `state.done=True`, so even
   the last actor missed `group_outcome_reward`. v2.1 settles the
   terminal-only rubrics (`group_outcome`, `hoarding_penalty`,
   `infected_deception`) for ALL agents at the moment the episode ends.

Plus: `obs.metadata` now exposes `last_actor_id`, `cumulative_rewards`
per agent, and `raw_reward` (un-clipped) so training code can pick the
right field instead of the OpenEnv-clipped `obs.reward`.

## Layout

```
v2/
├── survivecity_v2_env/   # env code (10 modules)
├── training/             # train.py (DGX 30GB), eval.py (15GB), simulator.py
├── server/               # FastAPI app on port 7861
├── tests/                # pytest
├── checkpoints/          # local LoRA checkpoints (gitignored)
├── eval_results/         # eval_step_NNNN.json + plots (gitignored)
└── results/              # simulator transcripts (gitignored)
```

## Install

From the repo root:

```bash
pip install -e v2/[train]            # for DGX training
pip install -e v2/                   # env + server only (15GB eval host)
pip install -e v2/[train,unsloth]    # if you want Unsloth fast kernels (Ampere+ only)
```

## Run on a Hugging Face Space (A10G / 24 GB)

This repo is set up to be deployed as a **Docker SDK Space** with GPU
hardware. The YAML front-matter at the top of this README is what HF
reads to configure the Space (`sdk: docker`, `dockerfile: Dockerfile.dgx`,
`app_port: 7860`, suggested hardware `a10g-large`).

### One-time setup on your Space

1. Create a Space (Docker SDK) or duplicate this repo into one:
   ```bash
   git clone <this-repo>
   cd zombiee-v2
   git remote add space https://huggingface.co/spaces/<YOUR-USERNAME>/<SPACE-NAME>
   git push space main
   ```
2. In the Space's **Settings → Secrets**, add:
   - `HUGGINGFACE_TOKEN` — a write-scoped token from
     https://huggingface.co/settings/tokens. Do **not** hard-code it
     anywhere in the repo. The container reads it from `os.environ` at
     runtime.
   - `HUB_MODEL_ID` (optional) — e.g. `YOUR-USERNAME/zombiee-v2`.
     If unset, the container falls back to `<space-owner>/zombiee-v2`
     using the `SPACE_AUTHOR_NAME` env var that HF injects automatically.
3. In **Settings → Hardware**, pick **Nvidia A10G large** ($1.50/hr,
   12 vCPU, 46 GB RAM, 24 GB VRAM). The "small" tier (15 GB system RAM)
   works but is tight when datasets / metrics buffers / model load
   coincide. The extra $0.50/hr buys serious peace of mind.
4. Click **Restart this Space**. The container will:
   - Boot in ~2 min (image pulls + smoke test).
   - Start a tiny FastAPI status sidecar on port 7860 so the Space UI
     shows "Running" — this is purely cosmetic, the actual workload is
     the training process running in the foreground.
   - Run training with v2.1 24 GB-safe defaults
     (`--num-generations 8 --max-completion-length 384 --rollout-limit 60`,
     4-bit nf4 quant, gradient checkpointing, LoRA r=32 α=64).
   - Push every checkpoint + the metrics file + plots to your Hub
     model repo, plus a heartbeat metrics push every 5 min.
5. Watch the **Logs** tab for `[progress] step N/15` and the per-call
   reward stats. Plots will appear at `<HUB_MODEL_ID>/plots/*.png` after
   training completes.

### Cost / time estimate (A10G large, $1.50/hr)

| What | Time | Cost |
|---|---|---|
| 1-step smoke test | ~10 min | ~$0.25 |
| **15-step run (recommended)** | **~2-2.5 h** | **~$3-3.75** |
| 20-step run | ~2.5-3.5 h | ~$3.75-5.25 |

If your A10G run hits OOM, override the CMD via Space "Variables"
(`EXTRA_ARGS=--num-generations 6 --max-completion-length 320`) — the
Dockerfile reads `EXTRA_ARGS` and appends it to the python command.

### Stopping the Space cleanly

When the 15 steps finish, the training process exits with code 0 and the
container exits — HF stops billing within a minute. If you want to abort
mid-run, click **Pause Space** in the UI; partial checkpoints + metrics
through the last save_step are already on the Hub repo.

## Train on a 24 GB GPU (bare-metal / Colab / Kaggle)

The v2.1 default flags fit a 24 GB box (`num_generations=8`,
`max_completion_length=384`, 4-bit nf4 base, gradient checkpointing on).
Bare-metal usage:

```bash
cd v2
pip install -e .[train]
python -m training.train \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --max-steps 15 \
    --save-steps 1 --save-total-limit 15 \
    --output-dir ./checkpoints \
    --rollout-limit 60 \
    --push-to-hub --hub-model-id noanya/zombiee-v2 \
    --resume-from-checkpoint auto
```

Override any default for a 30 GB DGX:
`--num-generations 12 --max-completion-length 512 --no-4bit`.

## Train on DGX (30 GB VRAM)

The current `Dockerfile.dgx` is sized for 24 GB by default (so it works
on HF Spaces A10G). On a 30 GB DGX you can EITHER:

- Use the same `Dockerfile.dgx` and override the CMD at run-time:
  `EXTRA_ARGS="--num-generations 12 --max-completion-length 512 --no-4bit"`.
- Or check out `Dockerfile.dgx.bak` — the original DGX-validated
  container (April 2026) with the 30 GB defaults baked in. Same
  dep-install logic, just different runtime flags.

### Containerised (recommended — uses the DGX-tested pin set)

```bash
docker build -f Dockerfile.dgx -t survivecity-v2-dgx .

# Run training. Mount checkpoints/ and eval_results/ as volumes so artefacts
# survive container restarts. Pass HUGGINGFACE_TOKEN if pushing to Hub.
docker run --rm --gpus all \
    -e HUGGINGFACE_TOKEN=$HF_TOKEN \
    -v "$(pwd)/v2/checkpoints:/app/checkpoints" \
    -v "$(pwd)/v2/eval_results:/app/eval_results" \
    survivecity-v2-dgx
```

The container's default CMD fits a 12-hour DGX session: **15 steps, checkpoint
every step (15 saves total), num_generations=12, LoRA r=32 α=64,
max_completion_length=512, gradient checkpointing on, bf16 base (--no-4bit),
adamw_torch_fused optimizer, VRAM holder reserving the unused 14 GB so
co-tenants on a shared DGX can't claim it mid-run** (`--vram-reserve-gb 16`
by default; set 0 if you own the GPU exclusively). Override the CMD to push
to Hub:

```bash
docker run --rm --gpus all \
    -e HUGGINGFACE_TOKEN=$HF_TOKEN \
    -v "$(pwd)/v2/checkpoints:/app/checkpoints" \
    survivecity-v2-dgx \
    python -m training.train \
        --max-steps 15 --save-steps 1 --save-total-limit 15 \
        --num-generations 12 --grad-accum-steps 4 \
        --max-completion-length 512 \
        --lora-r 32 --lora-alpha 64 \
        --no-4bit \
        --push-to-hub --hub-model-id noanya/zombiee-v2 \
        --resume-from-checkpoint auto
```

### Bare-metal (if your DGX is set up directly)

```bash
cd v2
pip install -e .[train]            # pinned stack (see pyproject.toml [train])
pip install -e .[train,unsloth]    # add Unsloth fast kernels (Ampere+)
python -m training.train \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --max-steps 15 \
    --save-steps 1 \
    --save-total-limit 15 \
    --output-dir ./checkpoints \
    --num-generations 12 \
    --grad-accum-steps 4 \
    --max-completion-length 512 \
    --lora-r 32 --lora-alpha 64 \
    --max-seq-length 4096 \
    --no-4bit \
    --push-to-hub --hub-model-id noanya/zombiee-v2 \
    --resume-from-checkpoint auto
```

Checkpoints land in `checkpoints/checkpoint-{N}/` and are pushed to the Hub
after every save (so a 15GB box can pull and eval them mid-training).

## Metrics, plots, and safety pushes

Every `reward_fn` call writes one row to `<output-dir>/metrics.jsonl` (via
`training.metrics.MetricsLogger`). Each row captures parse rate, reward
mean/std/min/max, cumulative-reward stats, rollout length, the action
histogram, the **per-rubric average contribution** (so you can see which
of the 14 rubrics are firing positively vs negatively), and per-rollout
terminal outcomes (% healthy survived, % infected stayed hidden, %
episodes that reached each vote phase).

At end of training, the trainer also writes:

- `<output-dir>/train_summary.json` — aggregate stats across the run.
- `<output-dir>/plots/*.png` — 8 plot PNGs auto-generated from the JSONL
  (controlled by `--auto-plots` / `--no-auto-plots`):

  | File | What it shows |
  |---|---|
  | `reward_curve.png` | Composite reward (mean ± std) — the headline GRPO learning curve |
  | `cumulative_reward.png` | Agent-0 raw cumulative reward (un-clipped) over training |
  | `parse_rate.png` | parse_ok / n — should rise to ~100 % within a few calls |
  | `rollout_length.png` | Heuristic-rollout length (mean + max) per call |
  | `action_distribution.png` | Stacked-bar evolution of the model's first-action histogram |
  | `rubric_breakdown.png` | Two-panel stacked area of each rubric's positive/negative average contribution |
  | `survival_rates.png` | % healthy survived, % infected hidden, % episodes reaching each vote phase |
  | `eval_comparison.png` | Baseline vs trained, from `eval_results/eval_step_*.json` (only when `--eval-results-dir` given to the CLI) |

### Re-generate plots manually

```bash
python -m training.plots \
    --metrics-file ./checkpoints/metrics.jsonl \
    --eval-results-dir ./eval_results \
    --output-dir ./plots/
```

Only requires the lightweight `[plots]` extras (matplotlib + numpy +
huggingface_hub) — the eval/15GB box can run this without the full
training stack:

```bash
pip install -e v2/[plots]
```

### Heartbeat / safety push to HF Hub

`hub_strategy="every_save"` already pushes the entire `output_dir` on
every `save_steps`. With `--save-steps 1` that's "after every training
step" — but if a step **crashes** mid-execution, the metrics rows logged
during that step are lost.

The **`HubSafetyPushCallback`** adds a wallclock heartbeat that pushes
**only the metrics file** (small, ~tens of KB) every N minutes,
independent of save_steps. Default: every 5 minutes when `--push-to-hub`
is set. Tune with `--safety-push-every-min 10` (or `0` to disable).

```bash
export HUGGINGFACE_TOKEN=hf_...
python -m training.train \
    --push-to-hub --hub-model-id YOUR_USERNAME/zombiee-v2 \
    --safety-push-every-min 5 \
    ...
```

The HF Space at `https://huggingface.co/spaces/noanya/zombiee-v2`
hosts the env-only FastAPI server. Training runs on a separate GPU
box and pushes to a Hub model repo (e.g. `noanya/zombiee-v2`); the
two are intentionally separate so the Space can stay env-only and
the model repo can hold all checkpoints + metrics + plots.

## Eval on a separate 15 GB box (T4)

```bash
cd v2
python -m training.eval \
    --lora-path noanya/zombiee-v2 \
    --baseline-episodes 30 \
    --trained-episodes 10 \
    --eval-step 50 \
    --output-dir ./eval_results
```

The `--lora-path` accepts either a local checkpoint directory or a Hub repo
id (it will `snapshot_download` if the latter). Output goes to
`eval_results/eval_step_<N>.json` plus `eval_step_<N>_bars.png` (per-metric
bar chart) and an `eval_history.png` (cross-checkpoint trend, auto-updates
across runs).

To eval a specific in-progress checkpoint:

```bash
python -m training.eval --lora-path ./checkpoints/checkpoint-100 --eval-step 100
```

## Simulate one episode (rich text visualizer)

```bash
cd v2
python -m training.simulator \
    --lora-path ./checkpoints/checkpoint-100 \
    --seed 42 \
    --max-steps 100 \
    --output ./results/transcripts/sim_seed42.txt
```

Renders the full grid, all agent states, zombie positions, actions, rewards,
broadcasts, and phase changes (waves, votes, day/night, infection reveals)
each step. Use `--policy random` to compare against the baseline.

## Run server (OpenEnv compliance check)

```bash
cd v2
uvicorn server.app:app --host 0.0.0.0 --port 7861
# in another terminal
curl -s -X POST http://localhost:7861/reset -H 'Content-Type: application/json' -d '{"seed":42}' | head -c 200
```

## Run tests

```bash
cd v2
pytest -q
```

## What changed vs v1

| Aspect | v1 | v2 |
|---|---|---|
| Grid | 10×10 | **15×15** |
| Agents | 3 | **5** (A0..A4) |
| Infected | 1 (static) | **2** (biter + saboteur), both randomly assigned |
| Bite spread | none | **p=0.35** on adjacency (deterministic seeded RNG) |
| Resources | food only | **food + water + medicine** |
| Inventory | none (eat-on-pickup) | **3 slots/agent** |
| Voting | once (t=50) | **3 rounds** (t=30, 60, 90) |
| Broadcast | free | **noise meter** → zombies +1 step toward agents over threshold |
| Day/night | none | **day/night cycle** (t=0-24 day, 25-49 night, 50-74 day, 75-99 night) |
| Zombies | 3 fixed | start 3 + **waves at t=25/50/75** (+2/+3/+3, cap 12) |
| Action types | 8 | **14** (added drink, scan, pickup, drop, give, inject) |
| Reward rubrics | 3 | **14** (10 from v2.0 + 4 from v2.1: forage_shaping, zombie_proximity, anti_camp, infected_deception) |
| Damage credit | dropped for non-acting victims | **`pending_reward` accumulator** drained into `cumulative_rewards` at end-of-round |
| Terminal settlement | only last-actor saw `group_outcome` | **all agents** receive terminal-rubric credit on `done=True` |
| GRPO rollout | random actions (rewards floor-pinned) | **forage heuristic** (`reward_std` ~0.5 vs ~0.014 in v1) |

A v1 LoRA loaded onto v2 will never emit the new action types but still
produces parseable v2 actions — so zero-shot transfer is valid (just
suboptimal, which is the whole point of the transfer-evaluation experiment).
