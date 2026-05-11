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

# zombiee-v2 — Multi-Action GRPO on a 5-Agent Social Deduction Game

> Fine-tuning **Qwen2.5-3B** with **GRPO + LoRA** on a custom 5-agent zombie-survival environment. Built for the **Meta × Hugging Face × PyTorch OpenEnv AI Hackathon** (Bangalore, April 2026).

## Links

| | |
|---|---|
| 🎮 Live demo (v1 — failure-replay learning) | https://zombiee-tau.vercel.app |
| 📦 v1 repo (notebook prototype) | https://github.com/SirjanSingh/zombiee |
| 📦 v2 repo (this — full GRPO training pipeline) | https://github.com/SirjanSingh/zombiee-v2 |
| 🧪 Stack | TRL · PEFT · LoRA · Qwen2.5-3B · V100 fp16 · Docker on DGX |

> v2 has **no hosted demo** — it's a training pipeline, not a playable artifact. The v1 link above is a working failure-replay learning demo that gave this project its name.

## TL;DR

I fine-tuned Qwen2.5-3B on a 5-agent zombie social-deduction survival game using GRPO. **Across 4 documented training runs (~90 hours on a V100 DGX), the trained policy moved `mean alive at end` from 0.15 (random baseline) to 0.60 — a 4× improvement on the partial-credit metric.** Survival rate (binary: ≥1 healthy alive) stayed at 0% — but `mean alive at end`, `mean reward`, and action diversity all moved in the right direction. The full postmortem (including what I'd do differently) is below.

## Results — 4 runs across 90 hours

| run | change | trained alive/5 | trained reward | notes |
|---|---|---|---|---|
| 1 | initial GRPO (single-action, 100-step) | — | 0.879 | scan-spam (49.5%), 0% survival |
| 2 | scan-streak rubric (aborted @ 21) | — | — | scan-streak threshold of 2 unreachable in 1-action rollouts |
| 3 | flat per-scan penalty + forage-shaping at hunger≥1 | — | 0.719 | scan dropped 49% → 41%, still 0% survival |
| 4 | **multi-action GRPO (`--prefix-actions 5`)** | **0.60** vs 0.15 baseline | **1.109** vs 1.028 baseline | first measurable training win |

Pure-heuristic baseline (no LLM) at run 4 with the layout + heuristic fixes: `mean alive at end` 0.00 → 0.80, episode length lifted past the starvation cliff.

Action histogram trend across run 4 (run-1 → final):

| action | run 1 (step 100) | run 4 (step 60) |
|---|---|---|
| scan | 49.5% | **13.1%** |
| movement (L+R+U+D) | ~3% | **26.7%** |
| eat | <1% | 3.2% |
| vote_lockout | 1% | 11.4% |

## Key contributions

1. **Multi-action GRPO prefix rollouts** — TRL by default does one model action per completion; I extended it so the model emits a K-action JSON array applied across K env steps before heuristic takeover. Backward-compatible at K=1 (byte-identical to legacy); 4 new tests pin the invariant. Commits: `09a4717`, `48c0e4c`, `6e8fc43`.

2. **Reward-hacking forensics** — found and closed 3 distinct exploits across runs (scan-spam, vote-spam, idle-wait), each with a minimal rubric patch committed separately for clean revert (`1e814d4`, `4f34535`, `95b3d57`).

3. **Pure-heuristic baseline as an RL debugging methodology** — when survival stuck at 0% for three runs, I ran the env with no LLM at all and discovered the real bottleneck was starvation, not the policy. Layout expansion + heuristic threshold-4 forage fix moved baseline `mean alive at end` from 0.00 → 0.80 (`741f4b6`, `33e1df9`). The methodology generalizes: **always check the env+heuristic baseline before blaming the model**.

4. **Honest negative-result writeup** — documented why GRPO + 3B + single-agent + 15-component rubric hits a ceiling. The diagnosis (KL collapse, heuristic-driven A1-A4 dying regardless of A0's policy, sparse healthy-survival signal) is the contribution. The full memory directory is in `.claude/projects/.../memory/`.

## Postmortem — what I'd do differently

Three compounding limits made the survival ceiling hard to break:

- **GRPO isn't designed for 60-step trajectories.** It's a completion-level optimizer (math, code, RLHF preferences). Aggregating step-rewards into one scalar loses per-step credit assignment. **KL stayed at 0.000 across all 60 steps in run 4** — the smoking gun that the policy isn't even diverging from the reference. PPO with a value head is the right tool for trajectory RL.
- **The model controls only ~8% of actions.** Multi-action prefix bumped this to ~25% but the heuristic still dominates. Self-play multi-agent training would close this gap.
- **3B is small for social deduction.** Theory of mind + planning + temporal awareness in <100 GRPO steps from a 3B base is a stretch. **SFT warmstart from heuristic trajectories**, then GRPO from that warm start, is the recipe I'd try next.

The pipeline itself works — it needs a different recipe, not a different codebase.

---

# SurviveCity v2 — technical reference (v2.1 reward-fix revision)

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

**Post-run-2 fixes (commits `1e814d4`, `4f34535`, `95b3d57`):**
- **`parse_action` word-boundary fix** — Strategy 3 used `in text_lower` (substring), so
  `"scan"` matched inside `"scan_target"`. Fixed with `re.search(r"\bscan\b", ...)`.
  Also closes: `"wait"` in `"waiting"`, `"eat"` in `"death"`, `"drop"` in `"dropped"`.
- **`scan_economy_reward`** (rubric 15) — flat **-0.03** per-scan cost on any scan
  (streak ≥ 1), escalating **-0.05/streak** past streak 2. Closes the scan-spam exploit
  that caused scan=49.5% in run 1 and 66.1% in run 2 under single-action GRPO rollouts.
- **`forage_shaping_reward` threshold** — now fires at hunger≥1 / thirst≥1 (was ≥7),
  with magnitude scaled linearly by urgency. Eliminates the dead zone at episode start
  that let scan dominate when forage gradient was absent.

**Post-run-3 fix — multi-action GRPO (commits `d64ebb7`, `48c0e4c`, `09a4717`, `6e8fc43`):**
- **Root cause from runs 1–3:** all three runs hit **0% survival** in eval despite
  improving training-time scan% from 49.5% → 40.6%. The reason: GRPO applied **one
  model action** per rollout, then the forage heuristic took over for ~55 more steps.
  The model never trained on steps 1, 5, 10, 15… so even a perfect step-0 policy
  couldn't outrun a heuristic that dies at step 18.
- **`--prefix-actions K` flag** (default `K=1`, backward-compat) — model emits a JSON
  array of K actions in a single completion, applied at agent A0's next K turns; the
  heuristic handles intermediate agent-1..4 turns and post-prefix steps. Run 4
  recommendation: **K=5** with `--step1-weight 1.0` (composite is now summed across
  K actions, so the legacy 5.0 would 5× over-weight).
- **`parse_actions()`** in `training/inference.py` — JSON-array parser with single-object
  fallback (so K=1 still parses any legacy completion).
- **Multi-action prompt template** — `SYSTEM_PROMPT_TEMPLATE_MULTI` selected when
  `prefix_actions > 1`; tells the model to commit to a K-step plan with array-shaped
  examples.
- **Enhanced metrics line** — `[metrics] step N: ...` now logs `reward_std`,
  `completion_length`, and `kl` so we can diagnose run 3's `kl=0.000` anomaly
  (was `kl=0.17` peak in run 1; cause TBD).

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

## Training history

### Run 1 — baseline (2026-04-30, 100 steps)

**Hardware:** Tesla V100-SXM2-32GB (`sm_70`), fp16 only, no bf16, no `adamw_torch_fused`.

**Command:**
```
--lr 5e-6 --beta 0.1 --no-4bit --optim adamw_torch --max-steps 100
--save-steps 5 --save-total-limit 20 --per-device-batch-size 8
--num-generations 8 --grad-accum-steps 4 --max-completion-length 256
--lora-r 32 --lora-alpha 64
```

**Reward trajectory:** -0.92 (step 1) → ~-0.60 plateau → -0.51 (step 100). ~6 h wallclock (~3.6 min/step).

**Eval results** (`--baseline-episodes 20 --trained-episodes 10 --max-steps-per-episode 150`):

| metric              | baseline | ckpt-25 | ckpt-100 |
|---------------------|----------|---------|----------|
| survival rate       | 0%       | 0%      | 0%       |
| mean reward         | 1.064    | 0.879   | 0.857    |
| infection isolation | 85%      | 100%    | 100%     |

**Diagnosis:** Model learned voting (100% infection isolation) but not survival.
Root cause: `scan` was Pareto-better than every other action — no rubric penalised it.
Action histogram: `scan=49.5%`, `eat=1.8%`; agents starved by step 17.

---

### Run 2 — partial (2026-05-09, aborted at step 21)

Applied fixes from commit `1e814d4` (parse_action word-boundary, scan_streak field,
invalid-target scan reclassified as wait, `scan_economy_reward` streak penalty).

**Result:** scan_economy never fired — the streak threshold of 2 was unreachable
because GRPO takes exactly one model action per rollout (the rest is forage heuristic,
which never scans). `scan_streak` never exceeded 1. Cumulative scan rose to **66.1%**
(worse than run 1). Run aborted at step 21.

Action histogram (steps 0–21):

| action      | rate |
|-------------|------|
| scan        | 66.1% |
| drink       | 14.4% |
| wait         | 6.1% |
| inject       | 4.8% |
| vote_lockout | 3.5% |
| movement     | 2.1% |
| eat          | 0.7% |

---

### Fixes applied for run 3 (commits since run 2)

| Commit | Change |
|--------|--------|
| `4f34535` | `scan_economy_reward`: flat **-0.03** per-scan cost on any scan (streak ≥ 1), keeps -0.05/streak escalation past streak 2. Net: scan=-0.025 vs wait=+0.005 at step 0. |
| `95b3d57` | `forage_shaping_reward`: fires at **hunger≥1** (was ≥7), magnitude scaled linearly with urgency — eliminates the dead zone at episode start that let scan dominate. |

Also confirmed: vote schedule is `[30, 50, 70, 90]` (committed `b8c55eb`).

---

### Run 3 — single-action GRPO with rubric fixes (2026-05-10)

**Hyperparams:** identical to run 1.

**Action histogram trended down well:** scan 63% → 49% → 40.6% by step 40 (vs run 2 stuck at 66%).
But survival in eval was still 0% across ckpt-25 (mean reward 0.719 vs baseline 0.891) and other
checkpoints — episode length capped at ~18 steps in eval, same as runs 1/2.

**Diagnosis confirmed:** the model's step-0 action was the only thing being trained. The
`forage_heuristic_action` ran every step from 1 onward, dying at ~step 18 from starvation
because the heuristic doesn't eat fast enough. Even a perfect step-0 model couldn't beat
the heuristic ceiling.

**KL anomaly:** `kl=0.000` across all 40 steps shown (run 1 had `kl=0.17` peak). Suspicious;
new diagnostic logging added (`d64ebb7`) to confirm whether KL is rounding, the ref model
is mis-wired, or the policy genuinely isn't diverging.

---

### Run 4 — multi-action GRPO (planned)

**Launch command (DGX, after `git pull`):**
```bash
TS=$(date +%Y%m%d_%H%M%S)
docker run --rm --gpus '"device=0"' --shm-size=16g \
  -v "$PWD/checkpoints":/app/checkpoints \
  -v "$PWD/training":/app/training \
  -v "$PWD/survivecity_v2_env":/app/survivecity_v2_env \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" -e HUGGINGFACE_TOKEN="$HF_TOKEN" \
  -e PYTHONUNBUFFERED=1 \
  -e EXTRA_ARGS="--lr 5e-6 --beta 0.1 --no-4bit --optim adamw_torch --max-steps 60 --save-steps 5 --save-total-limit 12 --per-device-batch-size 8 --num-generations 8 --grad-accum-steps 4 --max-completion-length 512 --lora-r 32 --lora-alpha 64 --prefix-actions 5 --step1-weight 1.0" \
  survivecity-v2-dgx \
  2>&1 | tee logs/train100_${TS}.log
```

**Why these flags:**
- `--prefix-actions 5` — model commits to a 5-action plan; gradient signal flows through 5 model
  decisions, not just 1.
- `--step1-weight 1.0` — composite is now `1.0 * sum(model_step_raws) + cum0 + bonus`. The legacy
  5.0 would 5× over-weight when summing across 5 actions.
- `--max-steps 60` — each step does ~5× more env work (5 model actions per rollout instead of 1),
  keeps wallclock close to run 3's 6 h.
- `--max-completion-length 512` — JSON array of 5 actions is ~150 tokens; 512 leaves headroom
  for verbose plans.

**Stop criteria / what to watch:**

| step | signal | meaning |
|------|--------|---------|
| 5–10 | `eat%` and movement % rising | gradient is shaping survival behaviour |
| 5+ | `kl > 0` in `[metrics]` | policy diverging from ref (run 3 was stuck at 0) |
| 20 | episode length > 18 in eval | model is steering past heuristic ceiling — first real survival sign |

**Eval after run 4:** standard `training.eval` still uses single-action mode (K=1).
The trained LoRA weights should still help at eval-time even though training shaped a
5-action policy. If eval survival is still 0%, multi-action eval is the next follow-up
(reuses `parse_actions()` already on `master`).

### Run 5 — Phase 1 hyperparameter recalibration (planned, May 2026)

After runs 1-4 all hit 0% survival, a 2025-literature sweep (see
`.planning2/11_RESEARCH_FINDINGS_AND_REVISED_PLAN.md`) found our
hyperparameters were miscalibrated by **5-10× vs the published GiGPO
Qwen2.5-3B recipes** (verl-agent run_alfworld_lora.sh, arxiv 2505.10978).
Run 5 fixes the calibration first, before any algorithm change.

**Launch command (DGX, after `git pull`):**
```bash
TS=$(date +%Y%m%d_%H%M%S)
docker run --rm --gpus '"device=0"' --shm-size=16g \
  -v "$PWD/checkpoints":/app/checkpoints \
  -v "$PWD/training":/app/training \
  -v "$PWD/survivecity_v2_env":/app/survivecity_v2_env \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" -e HUGGINGFACE_TOKEN="$HF_TOKEN" \
  -e PYTHONUNBUFFERED=1 \
  -e EXTRA_ARGS="--no-4bit --optim adamw_torch --prefix-actions 5 --step1-weight 1.0 --push-to-hub" \
  survivecity-v2-dgx \
  2>&1 | tee logs/train_run5_${TS}.log
```

All Phase 1 hyperparameter changes (LR 1e-5 → 3e-6, beta 0.04 → 0.01,
max-steps → 60, grad-accum → 8, max-completion → 512, LoRA r → 64,
invalid-action-penalty 0.10) are now **defaults** in `training/train.py`,
so the EXTRA_ARGS only carries DGX-specific knobs (no-4bit, optim) and
the v2.2 multi-action toggles. To reproduce runs 1-4 exactly, see the
"recover the runs 1-4 budget" stanza in `training/train.py`'s docstring.

**Why these defaults (vs run 4):**

| knob | run 4 | run 5 | source |
|------|------|-------|--------|
| LR | 5e-6 | **3e-6** | GiGPO ALFWorld LoRA |
| beta | 0.1 | **0.01** | GiGPO Sokoban (kl_loss_coef=0.01) |
| max-steps | 60 | 60 (paper target 150) | GiGPO total_epochs=150 |
| grad-accum | 4 | **8** | GiGPO train_batch=16 |
| LoRA r/alpha | 32/64 | **64/128** | GiGPO ALFWorld LoRA |
| max-completion | 512 | 512 | unchanged |
| format reward | `+0.10 if parse_ok` | **`-0.10 if parse_fail`** | GiGPO use_invalid_action_penalty |

The format-reward sign flip is the subtle one: inside a GRPO group whose
members all parse correctly, a uniform +0.10 cancels in the group mean
→ zero gradient toward format. The penalty puts unparseable trajectories
BELOW the mean and yields a real negative advantage.

**Stop criteria / what to watch (same as run 4 plus one new line):**

| step | signal | meaning |
|------|--------|---------|
| 1 | `r_std > 0.1` in `reward_fn` log line | within-group variance survived the hyperparam change |
| 5+ | `kl > 0` in `[metrics]` line | **the key Phase 1 success signal** — runs 1-4 had kl=0 |
| 5-10 | `parse_ok` rate climbing | invalid-action-penalty is shaping format |
| 20 | episode length > 18 in eval | beyond heuristic ceiling |

**Eval after run 5:** standard
`python -m training.eval --lora-path ./checkpoints/checkpoint-60 --eval-step 60`.

**Go/no-go after run 5 (per plan 11 §5/Phase 1):**
- ✅ Phase 1 wins if `mean_alive_at_end ≥ 1.9` (heuristic + 0.3) on 30 eval episodes.
- ❌ Continue to Phase 2 (GiGPO algorithm port) if < 1.9 OR if kl stays
  pinned at 0 across all 60 steps.

---

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
| Reward rubrics | 3 | **15** (10 from v2.0 + 4 from v2.1: forage_shaping, zombie_proximity, anti_camp, infected_deception + scan_economy from post-run-2 fix) |
| Damage credit | dropped for non-acting victims | **`pending_reward` accumulator** drained into `cumulative_rewards` at end-of-round |
| Terminal settlement | only last-actor saw `group_outcome` | **all agents** receive terminal-rubric credit on `done=True` |
| GRPO rollout | random actions (rewards floor-pinned) | **forage heuristic** (`reward_std` ~0.5 vs ~0.014 in v1); v2.2 adds opt-in **K-action prefix** (`--prefix-actions K`) so the model trains on multi-step plans |

A v1 LoRA loaded onto v2 will never emit the new action types but still
produces parseable v2 actions — so zero-shot transfer is valid (just
suboptimal, which is the whole point of the transfer-evaluation experiment).
