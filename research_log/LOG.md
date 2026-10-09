# ZombieE research log

Running narrative of every experiment, newest at the bottom. Structured numbers live in
`experiments.jsonl` (one JSON object per run); raw per-episode data in `data/`; episode replays
for animation in `replays/`. Nothing here is overwritten; corrections get a new entry.

Earlier history (runs 1-4, April-July 2026) is summarised in the README and `.planning2/`.
Headline from that era: four GRPO runs on Qwen2.5-3B, 0% survival every time.

---

## 2026-10-04 — Is the game even winnable? (manager session)

**Question:** after four 0% runs, could *any* policy survive this env, or is the ceiling the problem?

**Method:** script all 5 agents with progressively smarter hand-written policies, 100 episodes
each, seed 42, CPU (about 3 s per 60 episodes). Scripts in `tools/probes/`.

| policy | episode length | A0 lifetime (steps) | reached t=100 | survival |
|---|---|---|---|---|
| current heuristic (what GRPO rolls out with) | 16.1 | 13.0 | 0% | 0% |
| BFS planner, avoids zombies | 23.6 | 13.2 | 0% | 0% |
| water-camp planner (stock water, hide in safehouse) | 66.8 | 54.2 | 0% | 0% |
| oracle (knows who is infected, votes them out) | 66.8 | 49.0 | 0% | 0% |

**Findings**
1. Smarter play helps a lot (A0 lives 4x longer with the camp planner), so the env does reward strategy.
2. But nothing reaches t=100, not even the oracle. Zombies park around the safehouse, waves add up
   to 11, and resupply becomes suicidal after ~t=40. Agents die of thirst + hunger.
3. Separately, every GRPO training prompt was the step-0 state, so the model never trained on
   mid-game situations.

**Decision:** stop RL runs until the game is rebalanced (target: heuristic 5-20%, oracle >= 60%),
add a real strategic objective (radio + extraction, see `.planning2/13_V3_MANAGER_PLAN.md`), and train on
mid-episode states.

Data: `data/2026-10-04_ceiling_probe_v2.json`, `experiments.jsonl` rows with
`experiment=ceiling_probe_v2_balance`.

---

## 2026-10-04 — Balance knobs pulled into one config (worker session, W1)

**What:** every difficulty number (hunger/thirst clocks, zombie chase, waves, respawns, bite and
reveal timing) moved from scattered constants into `BalanceConfig` (`survivecity_v2_env/balance.py`).
The original tuning is preset `v2.2`. This is plumbing so W2/W3 can sweep difficulty.

**Check it changed nothing:** hashed every step of 40 seeded episodes under the heuristic, random and
oracle policies before and after. All three hashes identical; probe numbers identical (heuristic ep
length 16.02, oracle 66.76, both 0% survival). A golden-hash test now guards the `v2.2` preset.

---

## 2026-10-04 15:24 — calibration `v2.2-baseline` (balance `v2.2`, overrides: none)

Reference numbers for the original tuning, recorded by the new calibrate.py (W2). Matches the manager's ceiling probes exactly. Note: the starting biter always dies of infection_progression at t=30 (bite_at_step=0 + 30-step rule), so mid-game bites are rare.

100 episodes, seed 42, git `a9430ae`, all 5 agents scripted by the policy. surv = >=1 healthy agent alive at t=100.

| policy | surv | reach_T | ep_len | A0_life | healthy_end | alive_end | bites | top death causes (all agents) |
|---|---|---|---|---|---|---|---|---|
| random | 0% | 0% | 16.7 | 13.0 | 0.00 | 0.24 | 0.00 | zombie_attack 177, hunger 162, thirst 137 |
| heuristic | 0% | 0% | 16.1 | 13.0 | 0.00 | 0.47 | 0.00 | zombie_attack 265, hunger 147, thirst 41 |
| camp | 0% | 0% | 66.8 | 54.2 | 0.00 | 0.05 | 0.00 | thirst 259, hunger 153, infection_progression 83 |
| oracle | 0% | 0% | 66.8 | 49.0 | 0.00 | 0.00 | 0.00 | thirst 227, hunger 220, infection_progression 48, zombie_attack 5 |

Data: `data/2026-10-04_calibrate_v2.2-baseline.json`, `experiments.jsonl` rows with `tag=v2.2-baseline`.

---

## 2026-10-04 15:50 — calibration `v3-rc1` (balance `v3-rc1`, overrides: none)

W3 candidate preset. Not the default: the current heuristic cannot survive in any balance tried (see the W3 finding entry below).

100 episodes, seed 42, git `ef8ba1c`, all 5 agents scripted by the policy. surv = >=1 healthy agent alive at t=100.

| policy | surv | reach_T | ep_len | A0_life | healthy_end | alive_end | bites | top death causes (all agents) |
|---|---|---|---|---|---|---|---|---|
| random | 0% | 0% | 26.9 | 21.8 | 0.00 | 0.11 | 0.00 | hunger 360, thirst 113, zombie_attack 16 |
| heuristic | 0% | 0% | 26.4 | 24.0 | 0.00 | 0.76 | 0.00 | hunger 312, thirst 73, zombie_attack 39 |
| camp | 73% | 91% | 99.0 | 92.8 | 0.90 | 1.45 | 0.03 | thirst 213, hunger 142 |
| oracle | 83% | 90% | 98.7 | 70.1 | 1.24 | 1.25 | 0.03 | hunger 225, thirst 146, zombie_attack 3, infection_progression 1 |

Data: `data/2026-10-04_calibrate_v3-rc1.json`, `experiments.jsonl` rows with `tag=v3-rc1`.

---

## 2026-10-04 15:53 — balance sweep `w3-combo1` (12 variants)

W3 knob sweep: clock rate x zombie_move_every x chase radius, starting infected exempt from infection_progression.

100 episodes per cell, seed 42, git `ef8ba1c`. Cells: survival / episode length / A0 lifetime / healthy alive at end.

| variant | random surv / ep_len / A0 / healthy | heuristic surv / ep_len / A0 / healthy | camp surv / ep_len / A0 / healthy | oracle surv / ep_len / A0 / healthy |
|---|---|---|---|---|
| rate 0.5 move_every 1 radius None | 0% / 25.0 / 16.6 / 0.00 | 0% / 26.8 / 19.8 / 0.00 | 98% / 99.8 / 99.2 / 2.20 | 99% / 99.9 / 72.3 / 2.00 |
| rate 0.5 move_every 1 radius 5 | 0% / 30.1 / 23.0 / 0.00 | 0% / 29.7 / 24.1 / 0.00 | 97% / 99.8 / 97.7 / 2.09 | 96% / 99.7 / 71.9 / 2.19 |
| rate 0.5 move_every 2 radius None | 0% / 31.1 / 24.4 / 0.00 | 0% / 30.8 / 25.0 / 0.00 | 96% / 99.7 / 97.5 / 1.96 | 93% / 99.5 / 72.2 / 2.18 |
| rate 0.5 move_every 2 radius 5 | 0% / 31.9 / 25.9 / 0.00 | 0% / 31.6 / 27.6 / 0.00 | 97% / 99.8 / 98.1 / 2.05 | 93% / 99.5 / 71.8 / 2.22 |
| rate 0.6 move_every 1 radius None | 0% / 22.9 / 15.8 / 0.00 | 0% / 24.9 / 18.1 / 0.00 | 83% / 99.2 / 94.9 / 1.00 | 67% / 99.2 / 69.6 / 0.78 |
| rate 0.6 move_every 1 radius 5 | 0% / 26.0 / 20.4 / 0.00 | 0% / 26.1 / 20.9 / 0.00 | 72% / 98.7 / 92.8 / 0.92 | 78% / 98.5 / 69.4 / 1.13 |
| rate 0.6 move_every 2 radius None | 0% / 26.3 / 21.3 / 0.00 | 0% / 25.9 / 21.7 / 0.00 | 73% / 99.3 / 93.3 / 0.86 | 93% / 99.2 / 70.8 / 1.34 |
| rate 0.6 move_every 2 radius 5 | 0% / 26.8 / 21.7 / 0.00 | 0% / 26.4 / 23.6 / 0.00 | 74% / 99.2 / 92.8 / 0.91 | 88% / 98.7 / 69.0 / 1.26 |
| rate 0.75 move_every 1 radius None | 0% / 19.9 / 14.6 / 0.00 | 0% / 20.1 / 15.3 / 0.00 | 0% / 83.4 / 77.7 / 0.00 | 0% / 83.4 / 59.7 / 0.00 |
| rate 0.75 move_every 1 radius 5 | 0% / 21.6 / 17.0 / 0.00 | 0% / 21.1 / 17.5 / 0.00 | 0% / 82.6 / 75.2 / 0.00 | 0% / 83.5 / 59.7 / 0.00 |
| rate 0.75 move_every 2 radius None | 0% / 21.9 / 17.5 / 0.00 | 0% / 21.6 / 18.8 / 0.00 | 0% / 83.4 / 74.9 / 0.00 | 0% / 85.5 / 59.4 / 0.00 |
| rate 0.75 move_every 2 radius 5 | 0% / 22.1 / 17.6 / 0.00 | 0% / 21.9 / 19.3 / 0.00 | 0% / 83.1 / 75.6 / 0.00 | 0% / 82.9 / 58.8 / 0.00 |

Data: `data/2026-10-04_sweep_w3-combo1.json`, grid `tools/sweeps/w3_combo1.json`.

---

## 2026-10-04 — Two design findings from the W3 rebalance (worker session)

**1. The traitor died of old age before it could bite anyone.** The "30 steps after a bite with no
medicine, you die" rule also applied to the two agents infected at t=0 (their bite step is 0). So the
starting biter was revealed at t=25 and killed by the rule at t=30, the same turn as the first vote.
Under the oracle this happened in 48/100 episodes, and bites per episode were 0.00 across 400 runs.
Fix (approved by manager): `starting_infected_progression=False` in v3; `v2.2` keeps the old rule.
Exempting them alone barely moves the oracle (episode length 66.8 -> 66.5, still 0%), because the
real wall is food and water.

**2. The rollout heuristic cannot survive in any balance.** Sweeping clocks, zombie speed, chase
radius, waves and HP, the GRPO rollout heuristic stays at 0% everywhere, even where planners hit 99%.
Traces show three bugs, not difficulty:
- it drinks forever once standing on a water cell (rule 1 has no thirst check), and starves there;
- it "eats" on a depleted food cell while a zombie attacks it;
- greedy stepping walks into the walls next to the inner-ring water and stays stuck for 20+ turns.
With the three bugs fixed it still gets 0-1%, because it walks straight into zombies. Adding a
one-step "don't step next to a zombie" rule gets 13% at the candidate balance, inside the 5-20% target.

| policy at `v3-rc1` (rate 0.6, shamblers, chase radius 4, starting infected exempt) | survival | episode length |
|---|---|---|
| random | 0% | 26.9 |
| current heuristic | 0% | 26.4 |
| heuristic with 3 bug fixes (prototype) | 0% | 60.0 |
| bug fixes + zombie avoidance (prototype) | 13% | 74.9 |
| camp planner | 73% | 99.0 |
| oracle | 83% | 98.7 |

Open decision (manager/user): keep the old heuristic as a frozen baseline and add a fixed
"heuristic v3" as the rollout/baseline policy, or relax the 5-20% target. `v3-rc1` is a preset,
not yet the default. Prototype code: `data/2026-10-04_w3_heuristic_prototypes.py`.

---

## 2026-10-04 17:14 — calibration `v3-rc1-final` (balance `v3-rc1`, overrides: none)

W3 final: v3-rc1 is now the default; heuristic_v3 is the baseline/rollout policy; prompts/rubric/postmortem follow the config.

100 episodes, seed 42, git `cd4120f`, all 5 agents scripted by the policy. surv = >=1 healthy agent alive at t=100.

| policy | surv | reach_T | ep_len | A0_life | healthy_end | alive_end | bites | top death causes (all agents) |
|---|---|---|---|---|---|---|---|---|
| random | 0% | 0% | 26.9 | 21.8 | 0.00 | 0.11 | 0.00 | hunger 360, thirst 113, zombie_attack 16 |
| heuristic_v2 | 0% | 0% | 26.4 | 24.0 | 0.00 | 0.76 | 0.00 | hunger 312, thirst 73, zombie_attack 39 |
| heuristic_v3 | 11% | 11% | 74.3 | 50.6 | 0.13 | 0.70 | 0.34 | hunger 184, thirst 149, zombie_attack 90, infection_progression 7 |
| camp | 73% | 91% | 99.0 | 92.8 | 0.90 | 1.45 | 0.03 | thirst 213, hunger 142 |
| oracle | 83% | 90% | 98.7 | 70.1 | 1.24 | 1.25 | 0.03 | hunger 225, thirst 146, zombie_attack 3, infection_progression 1 |

Data: `data/2026-10-04_calibrate_v3-rc1-final.json`, `experiments.jsonl` rows with `tag=v3-rc1-final`.

---

## 2026-10-04 17:14 — calibration `v2.2-heuristic_v3` (balance `v2.2`, overrides: none)

Before/after reference: the fixed heuristic under the old balance.

100 episodes, seed 42, git `cd4120f`, all 5 agents scripted by the policy. surv = >=1 healthy agent alive at t=100.

| policy | surv | reach_T | ep_len | A0_life | healthy_end | alive_end | bites | top death causes (all agents) |
|---|---|---|---|---|---|---|---|---|
| heuristic_v3 | 0% | 0% | 26.2 | 18.2 | 0.00 | 0.48 | 0.00 | zombie_attack 251, hunger 155, thirst 46 |

Data: `data/2026-10-04_calibrate_v2.2-heuristic_v3.json`, `experiments.jsonl` rows with `tag=v2.2-heuristic_v3`.

---

## 2026-10-04 — v3 balance is live (worker session, W3 done)

The default game is now `v3-rc1`, and the fixed `heuristic_v3` is both the baseline and the policy
GRPO rolls out with. Targets met on 100 episodes: heuristic_v3 11% (target 5-20%), oracle 83%
(target >= 60%), random 0%. The old heuristic still scores 0% here, and the new heuristic scores 0%
under the old balance, so the result needed both fixes. The model's prompt now states the real v3
rules and shows public bite events ("A2 bit A4 at t=31"). Write-up: `.planning2/14_V3_BALANCE.md`.
Tables: the `v3-rc1-final` and `v2.2-heuristic_v3` entries above.

## 2026-10-04 — DGX ready (manager)

Fresh env on lnmdgx1 (V100-32GB): torch 2.5.1+cu121, transformers 4.46.3, trl 0.15.2, Qwen2.5-3B-Instruct.
Smoke test, GPU 3: model loads + one generation in 10.4 s, 6.5 GB peak (fp16), game prompt = 1298 tokens.
Untrained Qwen's first move at t=0 (seed 7): `{"action_type": "scan", "scan_target": 2}`. The base
model's prior already leans to scanning, the same behaviour that became scan-spam in run 1.

---

## 2026-10-04 17:48 — Training on the moments that matter (W6 smoke, worker session)

Training prompts used to be the step-0 state every time. Now each prompt is A0's real view at a
mid-game moment: the game is played forward by a mix of planner, heuristic and noisy-planner
teammates, and the moment is picked for "decision density" (thirsty, hungry, hurt, outside, zombie
near, vote, bitten, bite just happened). Calm moments are capped at 20%. A0 is always healthy in
training. The reward replays that exact moment, plays the model's 5 actions, and lets the camp
planner continue for 25 steps.

**Dataset the DGX run will use** (200 prompts, seed 42): time histogram 0-9: 51, 10-19: 21, 20-29: 20, 30-39: 25, 40-49: 23, 50-59: 16, 60-69: 16, 70-79: 10, 80-89: 12, 90-99: 6.
Tags: hungry 116, zombie_near 77, outside 65, vote 39, thirsty 35, hurt 17, bitten 7, bite_recent 6. Routine share 20%.

**Reward smoke** (16 scenarios, real reward_fn, fake completions; mean reward):

| completion | shaped window (step1_weight 1) | survival window (step1_weight 0) |
|---|---|---|
| planner's own 5 actions | -2.049 | +0.531 |
| 5 x wait | -1.778 | +0.406 |
| garbage (unparseable) | -1.878 | +0.306 |
| planner beats / loses to wait | 3/16 / 11/16 | 1/16 / 0/16 |

Finding: the shaped window return (the env's 15 rubrics summed over the window) prefers waiting to
the planner's moves, and in a 64-state probe it ranked a random A0 (-1.15) above the camp planner
(-2.53) although the planner kept A0 alive in 42/64 windows vs 27/64. Dying ends the per-step
hunger/thirst penalties, so an early death reads as cheaper than surviving hungry. The survival
window return (+1 alive / -1 dead, small HP and newly-infected terms) ranks them the right way.
Recommendation for the first DGX run: `--window-return survival --step1-weight 0`. Data: `data/2026-10-04_w6_smoke.json`.

## 2026-10-04 — Pre-launch checks + run 6 (v3) launched (manager)

**Reward sparsity probe** (24 v3-rc1 training states, 6 candidate 5-action plans each: camp planner,
heuristic_v3, all-wait, 3x random; survival window return, step1_weight 0):

| horizon H | states where plans score differently | spread > 0.25 | mean std |
|---|---|---|---|
| 25 | 11/24 | 10/24 | 0.432 |
| 40 | 10/24 | 10/24 | 0.456 |
| 60 | 10/24 | 10/24 | 0.451 |

About 45% of GRPO groups carry signal, and a longer window does not add any, so H=25 stays (cheaper).
The worker's "1/16" figure compared planner vs wait only; real samples are more diverse.

**Cross-machine determinism:** the v2.2 golden hash differed between Windows (Python 3.13) and the
DGX (3.11). Game states hash identically on both, and rewards match to 9 decimals; only the last
float bits differ (Python 3.12 changed float `sum()`). The test now rounds rewards to 9 dp.
123/123 tests pass on the DGX.

**Run 6 (v3) launched** on lnmdgx1 GPU 3, commit fd25f31: GiGPO, K=5, mid-episode scenarios (200),
camp-planner rollout, survival window H=25, step1_weight 0, A0 healthy, balance v3-rc1, 60 steps,
fp16 / adamw_torch, LoRA defaults from Phase 1 (r64). Log: `~/zombiee-v3/logs/train_run6_v3_.log`.

## 2026-10-04 19:30 — Run 6 attempt 1 aborted (zero-variance groups); run 6b launched (manager)

**Attempt 1** (`--window-return survival`): crashed once on launch (README command lacked
`--per-device-batch-size 8`; TRL needs the global batch divisible by num_generations=8). After the
fix: parse 8/8, varied actions, but **5/5 reward calls had std = 0**. Every group of 8 either all
died in the window (-1.0) or all survived at full HP (+1.5), so GRPO had no gradient. Stopped
before any checkpoint.

**Fix:** `--window-return graded`. Same survival objective with partial credit:
dead = -1 + 0.5 x fraction of the window survived; alive = +1 + 0.5 x hp + 0.25 x food headroom
+ 0.25 x water headroom (-0.5 if newly infected). Tie probe (24 v3-rc1 states, 8 random-policy plans
each, `tools/probes/tieprobe.py`, run on the DGX): zero-spread groups **7/24 survival -> 2/24 graded**.

**Run 6b** (commit 76e13ee): same config as attempt 1 but graded return, output
`checkpoints/run6b_v3`, log `logs/train_run6b_v3_20261004_192619.log`, GPU 3. First calls:
std 0.036 and 0.012 (non-zero). Speed: ~2 min per reward call x 8 per optimizer step, about 15 min
per step, so ~15 h for 60 steps. Checkpoints every 10 steps (~2.5 h).

Known cosmetic bug: after startup the training log stops printing INFO lines (no `reward_fn #N`
summaries). `checkpoints/run6b_v3/metrics.jsonl` has every call's stats, so nothing is lost.

## 2026-10-04 20:00 — Training was 5x slower than it should be: fixed; run 6c launched (manager)

Run 6b took ~550 s per optimizer step although completions average only 73 tokens. Cause: TRL 0.15.2
generates inside `_prepare_inputs` while the model is in train mode, and with gradient checkpointing
on, transformers silently forces `use_cache=False` (the warning was hidden by
`TRANSFORMERS_VERBOSITY=error`). Without a KV cache every new token re-reads the ~1200-token prompt.

Benchmark on the same V100 (Qwen2.5-3B + LoRA r64, 8 completions x 128 tokens):

| model mode during generate | time |
|---|---|
| train (what runs 1-6b did) | 188 s |
| eval (KV cache on) | 16 s |

Fix (commit 556dbf5): `run_in_eval_mode` wraps `_prepare_inputs` (generation + reference log-probs,
all no-grad), then restores train mode for the loss. Applied to both the GiGPO and plain GRPO trainers,
with tests. Run 6c: reward calls every ~12.5 s (was ~70 s), so ~2 min/step and ~2 h for 60 steps
instead of ~9 h. This also means every earlier run (1-4) paid this slowdown.

Run 6c = run 6b's config + the fix: GPU 3, `checkpoints/run6c_v3`, log `logs/train_run6c_v3_20261004_195959.log`.
Run 6b was stopped at ~step 3 (its metrics.jsonl kept).

---

## 2026-10-04 20:06 — eval `base-qwen` (closed loop, balance `v3-rc1`)

A0 driven by each policy for the whole episode (model re-plans every 5 A0 turns); A1-A4 = `camp`; same 30 seeds for every row; A0 healthy = True; git `e36a26b`; 67 s.

| A0 policy | A0 survives to end | A0 lifetime | team survival | healthy at end | parse | A0 outcome |
|---|---|---|---|---|---|---|
| base model | 0% | 20.8 | 63% | 0.73 | 100% | thirst 26, hunger 4 |
| heuristic_v3 | 3% | 63.6 | 60% | 0.70 | - | thirst 16, hunger 10, alive 1, zombie_attack 3 |
| camp | 0% | 94.5 | 67% | 0.77 | - | thirst 23, hunger 7 |
| random | 0% | 24.1 | 67% | 0.80 | - | hunger 21, thirst 7, zombie_attack 2 |
| wait | 0% | 24.0 | 63% | 0.73 | - | thirst 30 |

Data: `data/2026-10-04_eval_base-qwen.json`.

**Reading (manager):** A0 lifetime is the metric that separates policies; team survival (60-67% in
every row) is carried by the camp-planner teammates whoever plays A0. The untrained base model
(20.8) is worse than waiting every turn (24.0): it dies of thirst without fetching water. Targets
for training: heuristic_v3 63.6, camp planner 94.5. Even camp rarely reaches t=100 as A0: it stops
water sorties near the end and dies of thirst around t=94 (worth fixing before it becomes the
W7 SFT teacher).

---

## 2026-10-04 20:30 — eval `run6c-ckpt10` (closed loop, balance `v3-rc1`)

A0 driven by each policy for the whole episode (model re-plans every 5 A0 turns); A1-A4 = `camp`; same 30 seeds for every row; A0 healthy = True; git `a0f0c5c`; 64 s.

| A0 policy | A0 survives to end | A0 lifetime | team survival | healthy at end | parse | A0 outcome |
|---|---|---|---|---|---|---|
| model(checkpoint-10) | 0% | 20.8 | 63% | 0.73 | 100% | thirst 27, hunger 3 |

Data: `data/2026-10-04_eval_run6c-ckpt10.json`.

---

## 2026-10-04 20:57 — eval `sft-camp-v3` (closed loop, balance `v3-rc1`)

A0 driven by each policy for the whole episode (model re-plans every 5 A0 turns); A1-A4 = `camp`; same 30 seeds for every row; A0 healthy = True; git `0a12abb`; 63 s.

| A0 policy | A0 survives to end | A0 lifetime | team survival | healthy at end | parse | A0 outcome |
|---|---|---|---|---|---|---|
| model(sft_camp_v3) | 0% | 25.1 | 63% | 0.73 | 100% | hunger 29, thirst 1 |

Data: `data/2026-10-04_eval_sft-camp-v3.json`.

---

## 2026-10-04 21:36 — eval `sft-dagger1` (closed loop, balance `v3-rc1`)

A0 driven by each policy for the whole episode (model re-plans every 5 A0 turns); A1-A4 = `camp`; same 30 seeds for every row; A0 healthy = True; git `1c35c1b`; 92 s.

| A0 policy | A0 survives to end | A0 lifetime | team survival | healthy at end | parse | A0 outcome |
|---|---|---|---|---|---|---|
| model(sft_dagger1) | 0% | 33.6 | 63% | 0.73 | 100% | thirst 30 |

Data: `data/2026-10-04_eval_sft-dagger1.json`.

---

## 2026-10-04 22:07 — eval `run6c-ckpt30` (closed loop, balance `v3-rc1`)

A0 driven by each policy for the whole episode (model re-plans every 5 A0 turns); A1-A4 = `camp`; same 30 seeds for every row; A0 healthy = True; git `1c35c1b`; 64 s.

| A0 policy | A0 survives to end | A0 lifetime | team survival | healthy at end | parse | A0 outcome |
|---|---|---|---|---|---|---|
| model(checkpoint-30) | 0% | 21.5 | 63% | 0.73 | 100% | thirst 30 |

Data: `data/2026-10-04_eval_run6c-ckpt30.json`.

---

## 2026-10-04 22:08 — eval `run6c-ckpt60` (closed loop, balance `v3-rc1`)

A0 driven by each policy for the whole episode (model re-plans every 5 A0 turns); A1-A4 = `camp`; same 30 seeds for every row; A0 healthy = True; git `1c35c1b`; 65 s.

| A0 policy | A0 survives to end | A0 lifetime | team survival | healthy at end | parse | A0 outcome |
|---|---|---|---|---|---|---|
| model(checkpoint-60) | 0% | 21.7 | 63% | 0.73 | 100% | thirst 30 |

Data: `data/2026-10-04_eval_run6c-ckpt60.json`.

---

## 2026-10-04 22:26 — eval `run7-ckpt60` (closed loop, balance `v3-rc1`)

A0 driven by each policy for the whole episode (model re-plans every 5 A0 turns); A1-A4 = `camp`; same 30 seeds for every row; A0 healthy = True; git `1c35c1b`; 76 s.

| A0 policy | A0 survives to end | A0 lifetime | team survival | healthy at end | parse | A0 outcome |
|---|---|---|---|---|---|---|
| model(checkpoint-60) | 0% | 26.8 | 63% | 0.80 | 100% | hunger 28, thirst 2 |

Data: `data/2026-10-04_eval_run7-ckpt60.json`.

---

## 2026-10-04 22:30 — eval `sft-dagger2` (closed loop, balance `v3-rc1`)

A0 driven by each policy for the whole episode (model re-plans every 5 A0 turns); A1-A4 = `camp`; same 30 seeds for every row; A0 healthy = True; git `1c35c1b`; 167 s.

| A0 policy | A0 survives to end | A0 lifetime | team survival | healthy at end | parse | A0 outcome |
|---|---|---|---|---|---|---|
| model(sft_dagger2) | 70% | 99.7 | 97% | 1.47 | 100% | thirst 9, alive 21 |

Data: `data/2026-10-04_eval_sft-dagger2.json`.

---

## 2026-10-04 22:53 — eval `sft-dagger2-verify` (closed loop, balance `v3-rc1`)

A0 driven by each policy for the whole episode (model re-plans every 5 A0 turns); A1-A4 = `camp`; same 60 seeds for every row; A0 healthy = True; git `1c35c1b`; 361 s.

| A0 policy | A0 survives to end | A0 lifetime | team survival | healthy at end | parse | A0 outcome |
|---|---|---|---|---|---|---|
| model(sft_dagger2) | 72% | 99.0 | 92% | 1.43 | 100% | alive 43, thirst 14, hunger 3 |
| heuristic_v3 | 2% | 60.9 | 58% | 0.68 | - | hunger 19, thirst 28, zombie_attack 7, infection_progression 5, alive 1 |
| camp | 0% | 93.8 | 62% | 0.73 | - | thirst 41, hunger 19 |
| random | 0% | 23.6 | 68% | 0.80 | - | hunger 45, zombie_attack 4, thirst 11 |
| wait | 0% | 24.0 | 58% | 0.70 | - | thirst 60 |

Data: `data/2026-10-04_eval_sft-dagger2-verify.json`.

---

## 2026-10-04 23:00 — HEADLINE: DAgger makes Qwen survive the full game; RL alone barely moved (manager)

Closed-loop eval (`training/eval_v3.py`): our model plays A0 for the whole game, re-planning every 5 of
its turns; A1-A4 = camp planner; balance v3-rc1; A0 lifetime in turns.

| A0 played by | how it was trained | A0 lifetime | A0 alive at t=100 | team survival |
|---|---|---|---|---|
| untrained Qwen2.5-3B | none | 20.8 | 0% | 63% |
| run 6c ckpt-30 / ckpt-60 | GiGPO RL from scratch, 60 steps | 21.5 / 21.7 | 0% | 63% |
| SFT warm-start | imitate camp planner, 1845 states | 25.1 | 0% | 63% |
| run 7 ckpt-60 | GiGPO RL from the SFT model, 60 steps | 26.8 | 0% | 63% |
| DAgger round 1 | + 1229 states the model itself reached, teacher-labelled | 33.6 | 0% | 63% |
| **DAgger round 2** | + 1848 more own-states (4922 total) | **99.7** | **70%** | **97%** |
| *camp planner (teacher)* | hand-written | *94.5* | *0%* | *67%* |

(30 episodes, eval seed 1234.) **Verified on fresh seeds** (4321, 60 episodes, all baselines on the same
seeds): DAgger-2 model **72% alive at t=100 (43/60), lifetime 99.0, team survival 92%**; camp 0% / 93.8;
heuristic_v3 2% / 60.9; random 0% / 23.6; wait 0% / 24.0. Parse rate 100% throughout.

**What happened, in order**
1. Pure imitation (SFT) taught water runs (thirst deaths 26 -> 1) but every game ended at exactly t=25:
   the model drifted out of the safehouse (move_right 144 vs move_left 69) and starved outside. The
   teacher never eats; it survives hunger by staying inside, where healing cancels starvation damage.
   The model never saw how to recover from leaving: the classic compounding-error failure of behaviour
   cloning.
2. DAgger (Ross et al. 2011): run the student, label the states IT reaches with the teacher's 5-action
   plan (computed on a deepcopy of the live env), aggregate, retrain. Two rounds took ~90 min of one V100.
3. The student now beats its teacher. Overall action mixes are almost identical (wait 86% vs 84%,
   drink 5.0% vs 5.3%), so the gain is in timing, not style. Open question: the teacher stops water
   sorties near the end (dies of thirst ~t=94); the hypothesis is the student keeps re-planning water
   runs in the end game. To check with per-episode death-time and replay analysis.
4. RL with the current one-shot trainer (one 5-action plan from a mid-game state, scripted continuation,
   graded 25-step window return) barely moved the policy: run 6c 20.8 -> 21.7, run 7 25.1 -> 26.8.
   KL rose to ~0.02 only. This supports the research note: the RL needs the true closed-loop
   multi-turn trainer (run 8 plan) to matter.

Data: `data/2026-10-04_eval_{sft-camp-v3,sft-dagger1,sft-dagger2,sft-dagger2-verify,run6c-ckpt10,run6c-ckpt30,run6c-ckpt60,run7-ckpt60}.json`.
Adapters on the DGX: `checkpoints/sft_camp_v3`, `sft_dagger1`, `sft_dagger2` (best), `run6c_v3`, `run7_v3`.

---

## 2026-10-05 01:05 — CORRECTION: "student beats teacher" is a one-turn timing artifact (manager)

Replay analysis (`research_log/replays/ep-dagger2_*`, 8 matched seeds) of the camp planner vs the DAgger-2
model as A0:
- **Same strategy.** Both fetch water once (3 pickups by turn 6), drink 5 times, then stay in the
  safehouse for the rest of the game. Drink turns on seed 462141: camp [4, 7, 29, 51, 74], DAgger-2
  [4, 7, 29, 51, **75**].
- **One turn decides it.** Camp's last drink one turn earlier makes thirst reach 15 at t=98; it dies at
  t=99, one turn before the t=100 horizon (6 of 8 seeds). DAgger-2 peaks at thirst 14 and is alive at
  t=100. The "72% vs 0%" headline is real data but it measures a one-turn margin at the horizon,
  not a smarter strategy. A0 lifetime (99.0 vs 93.8) mostly reflects the same thing plus 2 camp
  hunger deaths (t=79, 87) after 13 outside turns.
- **What the model DID learn** (still true): from 20.8 turns (untrained, dies of thirst having never
  fetched water) to playing the planner's full survival routine, closed loop, with a 100% parse rate.
  That is the honest headline: imitation + DAgger taught a 3B LLM a complete survival routine.
- **Game-design finding:** in v3-rc1 "fetch 3 water, then hide" is a near-dominant strategy, so
  survival-to-t100 does not reward real decision-making. Radio + extraction (W4) is now the priority:
  it forces a late-game trip and makes timing, routing and deduction matter.

---

## 2026-10-05 02:15 — Run 8 (closed-loop RL) launched: GRPO vs GiGPO vs GAGPO (manager)

Trainer `training/train_closed_loop.py` (built by a subagent, branch run8-closed-loop, merged 9e7cf48; 144
tests pass on CPU). The model plays A0 at EVERY decision (re-plans every 5 A0 turns) for 30-turn windows
from mid-game start states; teammates = camp; graded return; per-decision credit by the chosen estimator;
PPO-clip + KL to the frozen DAgger-2 adapter; init = `checkpoints/sft_dagger2`.

DGX smoke (B=2, G=4, H=20, 2 steps, GPU 1): cuda, parse 1.00, KL 0 -> 0.0005 (frozen reference works on
peft 0.13.2), zero-variance groups 0.50 -> 0.00.

Full config (B=4 start states x G=8 rollouts, H=30, 60 steps, lr 3e-6, beta 0.01): step 1 = 150 decisions,
gen 47 s + update 78 s = 127 s/step, peak 10.8 GB on a 32 GB V100. ~2.2 h per estimator.
GPU 1: grpo then gigpo (z-score); GPU 5: gagpo (gamma 0.95, lambda 0.8). Each auto-evaluates its last
checkpoint with eval_v3 (30 eps, seed 1234; compare to DAgger-2: lifetime 99.7, alive at t=100 70%).

Caveat stated before results: v3-rc1 is near-solved by "fetch water, hide", and the DAgger-2 start policy
already survives most windows, so this run mostly validates the trainer; estimator differences may be
small. The informative comparison is on v3-rc2 (extraction) once W4 lands.

## 2026-10-05 02:43 — calibration `w4-v3-rc2` (balance `v3-rc2`, overrides: none)

W4 radio + extraction (v3-rc2 = v3-rc1 + radio t=60, helicopter t=90, corner zone r=2, max_steps 90). Headline = extraction rate.

100 episodes, seed 42, git `176b115`, all 5 agents scripted by the policy. surv = >=1 healthy agent alive at t=90. extract = >=1 healthy agent extracted and no infected aboard; failed_flight = an infected agent boarded; n_extr = mean healthy agents extracted.

| policy | extract | failed_flight | n_extr | surv | reach_T | ep_len | A0_life | healthy_end | alive_end | bites | top death causes (all agents) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 0% | 0% | 0.00 | 0% | 0% | 26.9 | 21.8 | 0.00 | 0.11 | 0.00 | hunger 360, thirst 113, zombie_attack 16 |
| heuristic_v2 | 0% | 0% | 0.00 | 0% | 0% | 26.4 | 24.0 | 0.00 | 0.76 | 0.00 | hunger 312, thirst 73, zombie_attack 39 |
| heuristic_v3 | 1% | 0% | 0.01 | 10% | 11% | 71.3 | 49.8 | 0.11 | 0.72 | 0.36 | hunger 189, thirst 138, zombie_attack 93, infection_progression 7 |
| camp | 33% | 14% | 0.49 | 49% | 50% | 85.9 | 75.3 | 0.77 | 1.32 | 0.70 | zombie_attack 150, hunger 135, thirst 41, infection_progression 33 |
| oracle | 54% | 4% | 0.78 | 63% | 63% | 87.3 | 66.4 | 0.91 | 1.03 | 0.34 | zombie_attack 217, hunger 123, thirst 38, infection_progression 17 |

Data: `data/2026-10-05_calibrate_w4-v3-rc2.json`, `experiments.jsonl` rows with `tag=w4-v3-rc2`.

---

## 2026-10-05 W4: radio + extraction, preset v3-rc2 (worker, branch `w4-extraction`)

**Rule.** v3-rc2 = v3-rc1 + extraction, `max_steps=90`. At t=60 a radio names one of 4 corner zones
(walkable cells within Chebyshev 2 of a corner, 8 cells each), picked by `random.Random(f"radio|{seed}")`,
so the episode rng stream is untouched (test: same seed, same episode up to the radio with extraction
on or off). At t=90 the helicopter boards every alive agent in the zone and the game ends; any infected
agent aboard (latent or revealed) fails the flight. Rewards (config weights, paid through
`pending_reward` into `cumulative_rewards`): healthy extracted +3.0 plus +1.0 per other healthy agent
extracted; healthy aboard a failed flight -1.0; infected agents (alive or dead) +3.0 if the flight fails
or nobody healthy gets out (including episodes that end early); +0.3 to each healthy agent alive at
t=30 and t=60. Prompt: "Survive; a radio message will come at t=60." before the radio, then
"RADIO: extraction at the NE corner (rows 0-2, cols 12-14), helicopter loads at t=90, N turns left.
Anyone infected aboard dooms the flight." plus one EXTRACTION rule line in the system prompt.
v2.2 and v3-rc1 trajectories, prompts, descriptions and rewards are byte-identical (hash-pinned).

**Headline (100 eps, seed 42, all 5 agents scripted, infected agents also head for the zone):**

| policy | target | extraction | failed flight | healthy extracted / ep | healthy alive at t=90 |
|---|---|---|---|---|---|
| random | ~0% | 0% | 0% | 0.00 | 0% |
| heuristic_v3 | 5-20% | **1%** (missed) | 0% | 0.01 | 10% |
| camp | between | 33% | 14% | 0.49 | 49% |
| oracle | >= 60% | **54%** (missed) | 4% | 0.78 | 63% |

**What it took on the policy side.** The first camp/oracle versions extracted 17%/27% (30 eps):
1. The plain camp planner never eats. Inside the safehouse healing cancels starvation, so it sits at
   hunger 30+ all game, and any trip outside then costs 1 HP per turn. With extraction on (only then:
   v3-rc1 behaviour is unchanged), camp carries one food item as the meal for the run, eats it when it
   leaves, keeps 2 water, and only starts supply sorties it can finish before hunger reaches 15.
   Ordering mattered: an early version sortied for food ahead of drinking and died of thirst.
2. The run: leave when turns left <= path length + slack; take a zombie-free route only if it is
   at most 2 steps longer, else the short route when the next cell is safe, else wait while there is
   time. Before this, BFS-around-zombies routes sent agents the long way round the map and they arrived
   late. Short-way fallback + eating the meal before a starving water run: oracle 13% -> 43% (60 eps);
   bounded detour: 47%. Slack sweep (60 eps, oracle): 3 -> 32%, 6 -> 47%, 8 -> 52%,
   10 -> 53%, 12 -> 58%, 16 -> 42%; set to 10.
3. heuristic_v3 now leaves at distance + 8 instead of walking to the zone at t=60 and starving there
   for 30 turns. Its slack does not matter (2/4/8/14/30 -> 2/2/1/1/0%): it is bounded by surviving to
   t=60 at all (19/100 episodes have no healthy agent left at the radio, 10% have one alive at t=90).

**Game knobs tried (100 eps unless noted), none adopted, preset stays as specified:**
- radio_step 50: oracle 44%, camp 35% (radio 45 on an earlier policy version, 30 eps: no gain either).
  More notice does not help; supplies and zombies are the limit.
- zone radius 3 (5x5): oracle 56%, camp 36% (vs 54/33). Within noise.
- no t=75 wave: oracle 57%, camp 33%. Within noise. Removing every wave gave oracle 63% (earlier policy version, 30 eps), but that
  removes most of the late-game pressure the objective is meant to add.

**What binds the oracle (fate of the 260 healthy agents alive at the radio, 100 eps):** extracted 78;
killed by zombies after leaving the safehouse 97; thirst 35 and hunger 32 after leaving; alive but not
in the zone 10; aboard a failed flight 3. At the radio 247/260 carry no water and 202/260 are starving
(hunger >= 15, harmless only inside). So the run starts with no supplies through a map holding 8-11
zombies. Infected agents are rarely the problem (failed flight 4%): most die before t=90 (infected
hunger x1.5). Next levers if the 60% bar matters: a planner that stocks water in the t=40-60 window
while hunger still allows sorties, or zombie-aware routing that predicts shambler moves. heuristic_v3
needs a better survival core, not a better extraction rule; or accept ~1% as the "weak baseline".

---

## 2026-10-05 03:30 — W4 accepted as v3-rc2; run 8 mid-way flat; v3-rc2 imitation pipeline queued (manager)

- **v3-rc2 accepted** (subagent, branch w4-extraction, merged d9cdbc6; 157 tests pass): extraction rate
  random 0%, heuristic_v3 1%, camp 33%, oracle 54% (target was 60%; stopped after principled fixes). Accepted
  because the goal was met: hiding no longer wins (0%), and the camp-to-oracle gap leaves room to learn.
  Bottleneck found by the subagent: at the radio, 247/260 healthy agents carry no water and 202 are starving,
  so the run starts unsupplied. Stocking up before t=60 is the strategy a learned policy could find.
- **Run 8 at step ~30 of 60 (v3-rc1):** return grpo 0.68 -> 0.56, gagpo 0.71 -> 0.71 (first 10 vs last 10
  steps), KL ~0.001, zero-variance groups 10-14%. Flat, as predicted for a near-ceiling start on an easy game.
- **Queued on GPU 5 after run 8's gagpo chain:** v3-rc2 imitation pipeline: camp-teacher SFT (2000 states) ->
  2 DAgger rounds -> eval on 60 fresh seeds vs camp/heuristic_v3/wait/random, recording 4 replays per row.
  Then run 8 (closed-loop RL) on v3-rc2 from that model: the experiment where RL has room to matter.

## 2026-10-05 04:17 — eval `run8-grpo` (closed loop, balance `v3-rc1`)

A0 driven by each policy for the whole episode (model re-plans every 5 A0 turns); A1-A4 = `camp`; same 30 seeds for every row; A0 healthy = True; git `d9cdbc6`; 177 s.

| A0 policy | A0 survives to end | A0 lifetime | team survival | healthy at end | parse | A0 outcome |
|---|---|---|---|---|---|---|
| model(checkpoint-60) | 0% | 98.5 | 67% | 0.77 | 100% | thirst 30 |

Data: `data/2026-10-05_eval_run8-grpo.json`.

---

## 2026-10-05 04:23 — eval `run8-gagpo` (closed loop, balance `v3-rc1`)

A0 driven by each policy for the whole episode (model re-plans every 5 A0 turns); A1-A4 = `camp`; same 30 seeds for every row; A0 healthy = True; git `d9cdbc6`; 181 s.

| A0 policy | A0 survives to end | A0 lifetime | team survival | healthy at end | parse | A0 outcome |
|---|---|---|---|---|---|---|
| model(checkpoint-60) | 0% | 97.0 | 63% | 0.73 | 100% | thirst 30 |

Data: `data/2026-10-05_eval_run8-gagpo.json`.

---

## 2026-10-05 04:49 — eval `rc2-sft` (closed loop, balance `v3-rc2`)

A0 driven by each policy for the whole episode (model re-plans every 5 A0 turns); A1-A4 = `camp`; same 30 seeds for every row; A0 healthy = True; git `d9cdbc6`; 65 s.

| A0 policy | extraction | A0 extracted | failed flight | A0 survives to end | A0 lifetime | team survival | healthy at end | parse | A0 outcome |
|---|---|---|---|---|---|---|---|---|---|
| model(sft_camp_rc2) | 23% | 0% | 7% | 0% | 17.9 | 33% | 0.47 | 100% | zombie_attack 26, hunger 4 |

Data: `data/2026-10-05_eval_rc2-sft.json`.

---

## 2026-10-05 06:08 — eval `rc2-dagger2` (closed loop, balance `v3-rc2`)

A0 driven by each policy for the whole episode (model re-plans every 5 A0 turns); A1-A4 = `camp`; same 60 seeds for every row; A0 healthy = True; git `d9cdbc6`; 204 s.

| A0 policy | extraction | A0 extracted | failed flight | A0 survives to end | A0 lifetime | team survival | healthy at end | parse | A0 outcome |
|---|---|---|---|---|---|---|---|---|---|
| model(sft_dagger_rc2_r2) | 18% | 0% | 5% | 0% | 31.0 | 30% | 0.35 | 100% | hunger 38, zombie_attack 21, thirst 1 |
| camp | 28% | 13% | 7% | 22% | 82.1 | 38% | 0.55 | - | zombie_attack 23, alive 20, thirst 14, infection_progression 1, hunger 2 |
| heuristic_v3 | 37% | 0% | 8% | 2% | 56.2 | 58% | 0.75 | - | hunger 22, thirst 29, zombie_attack 7, alive 2 |
| wait | 18% | 0% | 12% | 0% | 24.0 | 33% | 0.45 | - | thirst 60 |
| random | 22% | 0% | 5% | 0% | 23.6 | 32% | 0.45 | - | hunger 44, zombie_attack 5, thirst 11 |

Data: `data/2026-10-05_eval_rc2-dagger2.json`.

---

## 2026-10-05 06:27 — eval `run8-gigpo` (closed loop, balance `v3-rc1`)

A0 driven by each policy for the whole episode (model re-plans every 5 A0 turns); A1-A4 = `camp`; same 30 seeds for every row; A0 healthy = True; git `d9cdbc6`; 176 s.

| A0 policy | A0 survives to end | A0 lifetime | team survival | healthy at end | parse | A0 outcome |
|---|---|---|---|---|---|---|
| model(checkpoint-60) | 0% | 99.0 | 67% | 0.77 | 100% | thirst 30 |

Data: `data/2026-10-05_eval_run8-gigpo.json`.

---

## 2026-10-05 12:00 — Results: run 8 (RL on v3-rc1) did not help; imitation on v3-rc2 not working yet (manager)

**Run 8, closed-loop RL from DAgger-2 on v3-rc1** (60 steps each, eval 30 eps seed 1234):

| model | A0 lifetime | A0 alive at t=100 |
|---|---|---|
| DAgger-2 (start) | 99.7 | 70% |
| + GRPO | 98.5 | 0% |
| + GiGPO (z-score) | 99.0 | 0% |
| + GAGPO (gamma 0.95, lambda 0.8) | 97.0 | 0% |

Training-window return drifted down (grpo 0.68 -> 0.45, gigpo 0.68 -> 0.56, gagpo 0.71 -> 0.49, first vs last 10
steps; start states differ per step, so this is noisy), KL stayed <= 0.007. The alive-at-t100 drop is the same
one-turn cliff seen before (all three die of thirst at t=99 again). Verdict: on a near-solved game RL had
nothing to find and lost the lucky timing. No evidence yet that any estimator beats another here.

**v3-rc2 (extraction) imitation pipeline** (camp teacher; headline must be A0's own outcome, because the
team extraction column is carried by the camp teammates whatever A0 does: wait 18%, random 22%):

| A0 played by | A0 extracted | A0 lifetime | eval |
|---|---|---|---|
| SFT on 1792 camp states | 0% | 17.9 | 30 eps |
| DAgger round 1 (as rollout policy) | 0% | 22.4 | 200 eps, T=0.7 |
| DAgger round 1 | 0% | 26.3 | 200 eps rollouts of r2 |
| DAgger round 2 | 0% | 31.0 | 60 eps, seed 4321 |
| camp planner (teacher) | 13% | 82.1 | 60 eps, seed 4321 |
| heuristic_v3 | 0% | 56.2 | same |

Replays (4 matched seeds): the teacher camps (~50 waits), carries 1 food, eats it once, dies to zombies on the
t~80 extraction run in 3/4 games. The student spends 25-45 turns outside (move_down x20 in one game), picks
food up but eats it 0-1 times, and starves outside at t=24-44. The prompt does show inventory and hunger
(checked), so it is an imitation failure: compounding error, the same drift-outside failure as rc1 SFT.
DAgger is improving it each round (17.9 -> 22.4 -> 26.3 -> 31.0) but rounds collect fewer states (920, 1067)
because the student dies early.

**Hypothesis for next run:** 5-action blind plans (K=5) are fine for "sit in the safehouse" (rc1) but drift
during rc2's trips around zombies. Next: rc2 DAgger with K=3 (re-plan more often), 300 rollout episodes per
round, 3 rounds.

---

## 2026-10-05 11:46 — eval `v22-unwinnable` (closed loop, balance `v2.2`)

A0 driven by each policy for the whole episode (model re-plans every 5 A0 turns); A1-A4 = `camp`; same 4 seeds for every row; A0 healthy = True; git `c1ffdfb`; 2 s.

| A0 policy | A0 survives to end | A0 lifetime | team survival | healthy at end | parse | A0 outcome |
|---|---|---|---|---|---|---|
| camp | 0% | 56.0 | 0% | 0.00 | - | thirst 1, hunger 3 |
| heuristic_v3 | 0% | 12.0 | 0% | 0.00 | - | zombie_attack 4 |
| wait | 0% | 15.0 | 0% | 0.00 | - | thirst 4 |

Data: `data/2026-10-05_eval_v22-unwinnable.json`.

---

## 2026-10-05 19:22 — eval `rc2-k3-r1` (closed loop, balance `v3-rc2`)

A0 driven by each policy for the whole episode (model re-plans every 3 A0 turns); A1-A4 = `camp`; same 30 seeds for every row; A0 healthy = True; git `d9cdbc6`; 121 s.

| A0 policy | extraction | A0 extracted | failed flight | A0 survives to end | A0 lifetime | team survival | healthy at end | parse | A0 outcome |
|---|---|---|---|---|---|---|---|---|---|
| model(sft_rc2_k3_r1) | 27% | 0% | 20% | 0% | 24.2 | 53% | 0.63 | 100% | hunger 25, zombie_attack 5 |

Data: `data/2026-10-05_eval_rc2-k3-r1.json`.

---

## 2026-10-05 21:31 — eval `rc2-k3-r2` (closed loop, balance `v3-rc2`)

A0 driven by each policy for the whole episode (model re-plans every 3 A0 turns); A1-A4 = `camp`; same 30 seeds for every row; A0 healthy = True; git `d9cdbc6`; 145 s.

| A0 policy | extraction | A0 extracted | failed flight | A0 survives to end | A0 lifetime | team survival | healthy at end | parse | A0 outcome |
|---|---|---|---|---|---|---|---|---|---|
| model(sft_rc2_k3_r2) | 30% | 0% | 13% | 0% | 27.1 | 47% | 0.53 | 100% | hunger 23, zombie_attack 1, thirst 6 |

Data: `data/2026-10-05_eval_rc2-k3-r2.json`.

---

## 2026-10-06 00:34 — eval `rc2-k3-r3` (closed loop, balance `v3-rc2`)

A0 driven by each policy for the whole episode (model re-plans every 3 A0 turns); A1-A4 = `camp`; same 30 seeds for every row; A0 healthy = True; git `d9cdbc6`; 331 s.

| A0 policy | extraction | A0 extracted | failed flight | A0 survives to end | A0 lifetime | team survival | healthy at end | parse | A0 outcome |
|---|---|---|---|---|---|---|---|---|---|
| model(sft_rc2_k3_r3) | 37% | 0% | 13% | 0% | 80.7 | 67% | 0.77 | 100% | thirst 29, alive 1 |

Data: `data/2026-10-06_eval_rc2-k3-r3.json`.

---

## 2026-10-06 00:45 — eval `rc2-k3-final` (closed loop, balance `v3-rc2`)

A0 driven by each policy for the whole episode (model re-plans every 3 A0 turns); A1-A4 = `camp`; same 60 seeds for every row; A0 healthy = True; git `d9cdbc6`; 675 s.

| A0 policy | extraction | A0 extracted | failed flight | A0 survives to end | A0 lifetime | team survival | healthy at end | parse | A0 outcome |
|---|---|---|---|---|---|---|---|---|---|
| model(sft_rc2_k3_r3) | 27% | 0% | 5% | 0% | 80.0 | 37% | 0.43 | 100% | thirst 54, alive 3, hunger 1, infection_progression 2 |
| camp | 28% | 13% | 7% | 22% | 82.1 | 38% | 0.55 | - | zombie_attack 23, alive 20, thirst 14, infection_progression 1, hunger 2 |
| heuristic_v3 | 37% | 0% | 8% | 2% | 56.2 | 58% | 0.75 | - | hunger 22, thirst 29, zombie_attack 7, alive 2 |

Data: `data/2026-10-06_eval_rc2-k3-final.json`.

---

## 2026-10-08 16:45 — Why K=3 DAgger stalls at lifetime 80: the late game was never labelled (manager)

**What the r3 student actually does.** In all 60 final-eval episodes `sft_rc2_k3_r3` never eats (0 `eat`
actions; camp: 73). Its moves are exactly 3 left / 3 right / 2 up / 2 down per episode: it replays camp's
15-turn opening (fetch 2 water + 1 food, walk into the safehouse), then waits and drinks. Turn by turn it is
identical to camp (seed 58523) until t=55, where it drinks one turn early; by t=78 it has no water left,
`drink` is a no-op, and it dies of thirst at t=80. Camp at that point eats (t=67), votes, and goes out.

**Why.** Round R labels the states reached by student R-1. Student r2 died around t=27, so the data r3
was trained on stops there. 85% of the 9,694 aggregated rows are t<30; only 491 are t>=60, where the
eat / water-refill / helicopter decisions live. Lifetime went 24 -> 29 -> 80 and each round pushes the
labelled horizon out, which is normal DAgger progress, not a capacity limit.

**Teacher ceiling.** The oracle (54% team extraction) reads hidden infection state, so it can't be the
teacher. Camp is the best public-observation teacher: A0 lifetime 82, A0 extracted 13%. Imitation aims
to match camp; RL (GAGPO) is for beating it afterwards.

**Decision.** Continue DAgger rounds 4-5 from `sft_rc2_k3_r3` (not GAGPO yet; RL gave nothing on rc1).
One change: each round trains on all t>=30 rows + a 2,500-row sample of t<30 rows (the openings are
near-duplicates), so SFT time stays flat. Script `~/rc2_k3_cont.sh`, log `logs/rc2_k3_cont.log`, GPU 2
(shared with a 2 GB job, so the waiter's util cap went 25% -> 60%). Started 16:44 IST, ~3.5 h per round.
Watch: `eat` appearing in A0's actions, A0 lifetime > 82, A0 extracted > 0.

---

## 2026-10-09 01:50 — Better teacher: camp's A0 dies on the extraction run; a fair lookahead beats it (manager)

CPU, v3-rc2, A0 healthy, teammates `camp`, seeds `random.Random(999)`.

**Where camp's A0 dies (200 eps):** 65 zombie_attack + 34 thirst, all on the extraction run (t70-88). Typical thirst
death: drinks its last water ~t56, can't restock (zombies near the depots), leaves at t70 dry at thirst ~8.
Departure slack sweep (10/15/20/25/30): 10 is best; leaving earlier trades zombie deaths for thirst/hunger.

**camp_v2** (`training/policies.py`, `water_detour=True`): with no water, budget and route the run via the best
water cell; at a water cell drink at thirst >= 11 even next to a zombie (a hit costs 1 of 3 HP). 200 eps:
A0 extracted 18.5% (camp 17.5%), thirst deaths 34 -> 27 but zombie deaths 68 -> 85. Rules alone hit a ceiling.

**Lookahead teacher** (`tools/probes/lookahead.py`): from t60, for each legal A0 action, play the rest of the
episode M=4 times with camp_v2 (A0) + camp (teammates) on a deep copy and take the best mean score
(1.0 extracted + 0.1 alive + 0.002 lifetime). `fair` redraws what A0 can't see in each rollout: episode_seed
(drives hash-deterministic bites/cues/scans, and the zone before the t60 radio) and the infection of teammates
not publicly seen biting.

| A0 policy | N | A0 extracted | A0 alive at end | A0 deaths |
|---|---|---|---|---|
| camp | 200 | 17.5% | ~37% | zombie 68, thirst 34, hunger 12 |
| camp_v2 | 200 | 18.5% | - | zombie 85, thirst 27, hunger 15 |
| lookahead fair | 100 | **24%** | 62% | hunger 19, thirst 13, infection 6 |
| lookahead unfair (sees hidden state) | 100 | 31% | 69% | hunger 20, thirst 6, infection 5 |

Zombie deaths go to 0. Most survivors still miss the helicopter (62% alive, 24% out): the alive bonus likely makes
it too cautious, and the camp_v2 continuation eats badly. +-4.3 pts SE at N=100: suggestive, not yet conclusive.
Next: tune score/M, confirm on 300 seeds, then use it as the DAgger teacher. Cost ~12 s/episode on 11 CPU cores.

---

## 2026-10-09 03:30 — Model-size comparison begins: tiny CNN (552k) DAgger vs camp; Laya queued (manager)

Sirjan's idea: put a small, fast decision model next to Qwen 3B (Qwen stays). Two entries added:

**Tiny CNN** (`training/tiny_policy.py`, 551,882 params): public observation as an 11-channel 15x15 grid
(walls, food, water, safehouse, zombies, extraction zone, A0, teammates, known biters, zombie halo,
distance-to-A0) + 12 scalars -> 10 actions, every turn, illegal actions masked. DAgger vs `camp`
(round 0 = cloning camp rollouts), retrained-in-place each round, 8 epochs, CPU only. Eval: 200 seeds
(`random.Random(999)`), teammates camp.

| round | states | train acc | A0 lifetime | A0 alive end | A0 extracted |
|---|---|---|---|---|---|
| 0 | 16,327 | 89.3% | 44.7 | 1.5% | 0% |
| 1 | 25,948 | 95.0% | 71.4 | 8% | 0% |
| 2 | 40,001 | 97.9% | 75.2 | 10.5% | 0.5% |
| 3 | 54,876 | 99.1% | 74.3 | 13% | 2.5% |
| 4 | 69,993 | 99.2% | 74.7 | 13% | 1% |
| 5 | 84,843 | 99.3% | 78.3 | 20% | **3%** |

~10 min per round on a laptop CPU. First trained policy with A0 extracted > 0 (Qwen K=3 r4: 0%, lifetime 77.8).
Still far under camp (17.5%, ~37% alive); train acc 99% vs weak eval = compounding small errors on the run.
Data: `data/2026-10-09_tiny_camp.jsonl`. Checkpoint `checkpoints/tiny_camp.pt` (local, git-ignored).

**Laya** (`training/laya_policy.py`, convaiinnovations/laya-typed-decisions, 421M ModernBERT, Apache-2.0):
one typed `choice` question over the same 10 actions per turn. State = env description (what Qwen reads)
+ compact MAP section (nearest water/food, safehouse/zone distance, per-move blocked/next-to-zombie),
~400 tokens (Laya context 1,024). Pipeline `tools/laya_dagger.sh`: zero-shot baseline, round 0 cloning
(300 camp eps), rounds 1-3 DAgger (150 eps), waits subsampled to 1/3 of rows, `laya.train.finetune` from
the base each round (3 epochs, fp16 AMP). Runs from `/tmp/23ucs715_laya` on the DGX (venv with
transformers 4.57.6 over the zombiee env's torch 2.5.1; home quota is full). Queued 03:18: every GPU had
26-32 GB in use (smoke fine-tune OOM'd at 5 GB free); waits for a GPU with >= 12 GB free.

---

## 2026-10-09 11:00 — Laya round 0 (pure cloning of camp) reaches 13.5% A0 extracted (manager, overnight)

Laya-typed-decisions (421M) fine-tuned once on 14,738 camp-labelled states (300 camp episodes, waits subsampled),
3 epochs, `laya.train.finetune`, fp16 on a shared V100 (attempts 1-2 OOM'd when neighbours' jobs grew; attempt 3 on
GPU 6 at ~8.8 GB, 07:46-09:29). Eval: same 200 seeds (`Random(999)`), teammates camp, greedy masked argmax.

| A0 policy | params | A0 extracted | A0 alive end | A0 lifetime | A0 deaths | ms/decision |
|---|---|---|---|---|---|---|
| camp (teacher) | - | 17.5% | ~37% | 82.0 | zombie 68, thirst 34, hunger 12 | <1 |
| Laya zero-shot (30 eps) | 421M | 0% | 0% | 24.0 | hunger 26, thirst 4 | 58.7 |
| **Laya r0 (cloning)** | 421M | **13.5%** | **34.5%** | **82.2** | thirst 92, zombie 18, hunger 10 | ~71 |
| tiny CNN r5 (DAgger) | 552k | 3% | 20% | 78.3 | zombie 70, thirst 65, hunger 14 | ~1-2 |
| Qwen 3B K=3 r4 (DAgger) | 3B | 0% | 0% | 77.8 | zombie 23, thirst 7 (30 eps) | ~1000 |

Laya after plain behaviour cloning is within 4 pts of its teacher and far ahead of both other students. Likely
reasons, untested: a per-turn decision (no K=3 open-loop plans), the MAP section (per-move zombie adjacency,
nearest supplies) that the CNN must infer from pixels and Qwen from coordinates, and a classifier head trained with
a calibrated proper-scoring loss. Remaining failure is thirst (92/200). DAgger round 1 (12,481 new states from
Laya's own rollouts, 22,214 train items) started 10:03 on GPU 3.
Data: `data/2026-10-09_laya_dagger.jsonl`.

---

## 2026-10-08 23:28 — eval `rc2-k3-r4` (closed loop, balance `v3-rc2`)

A0 driven by each policy for the whole episode (model re-plans every 3 A0 turns); A1-A4 = `camp`; same 30 seeds for every row; A0 healthy = True; git `367947e`; 199 s.

| A0 policy | extraction | A0 extracted | failed flight | A0 survives to end | A0 lifetime | team survival | healthy at end | parse | A0 outcome |
|---|---|---|---|---|---|---|---|---|---|
| model(sft_rc2_k3_r4) | 27% | 0% | 7% | 0% | 77.8 | 43% | 0.57 | 100% | zombie_attack 23, thirst 7 |

Data: `data/2026-10-08_eval_rc2-k3-r4.json`.

---

## 2026-10-09 11:47 — eval `rc2-k3-r5` (closed loop, balance `v3-rc2`)

A0 driven by each policy for the whole episode (model re-plans every 3 A0 turns); A1-A4 = `camp`; same 30 seeds for every row; A0 healthy = True; git `367947e`; 397 s.

| A0 policy | extraction | A0 extracted | failed flight | A0 survives to end | A0 lifetime | team survival | healthy at end | parse | A0 outcome |
|---|---|---|---|---|---|---|---|---|---|
| model(sft_rc2_k3_r5) | 27% | 0% | 13% | 0% | 78.2 | 43% | 0.50 | 100% | thirst 12, zombie_attack 16, alive 2 |

Data: `data/2026-10-09_eval_rc2-k3-r5.json`.

---

## 2026-10-09 12:00 — eval `rc2-k3-r5-final` (closed loop, balance `v3-rc2`)

A0 driven by each policy for the whole episode (model re-plans every 3 A0 turns); A1-A4 = `camp`; same 60 seeds for every row; A0 healthy = True; git `367947e`; 798 s.

| A0 policy | extraction | A0 extracted | failed flight | A0 survives to end | A0 lifetime | team survival | healthy at end | parse | A0 outcome |
|---|---|---|---|---|---|---|---|---|---|
| model(sft_rc2_k3_r5) | 30% | 0% | 7% | 0% | 77.5 | 38% | 0.47 | 100% | zombie_attack 33, thirst 22, alive 3, infection_progression 2 |
| camp | 28% | 13% | 7% | 22% | 82.1 | 38% | 0.55 | - | zombie_attack 23, alive 20, thirst 14, infection_progression 1, hunger 2 |
| heuristic_v3 | 37% | 0% | 8% | 2% | 56.2 | 58% | 0.75 | - | hunger 22, thirst 29, zombie_attack 7, alive 2 |

Data: `data/2026-10-09_eval_rc2-k3-r5-final.json`.

---

## 2026-10-09 12:30 — Qwen K=3 DAgger r4-r5: flat at 0% A0 extracted; Laya (421M) is far ahead (manager)

Rounds 4-5 (late-game rows up-weighted, see 2026-10-08 16:45) changed Qwen's behaviour (it eats, leaves for the
helicopter) but not the outcome: r5 final, 60 seeds 4321, A0 extracted 0% / lifetime 77.5 / zombie 33, thirst 22
(camp on the same seeds: 13% / 82.1). Each round costs ~4-5 h of shared V100. On the 200-seed table Laya r0 (pure
cloning, ~2 h) is at 13.5%. Recommendation: stop Qwen DAgger on camp labels here; the comparison video has Qwen r5
as the "big LLM" entry. If Qwen gets another try it should be per-turn (K=1) with the same MAP state section Laya
uses, to separate model size from interface.

---

## 2026-10-09 13:30 — Tiny CNN vs lookahead teacher, round 0 (DGX CPUs) (manager)

Rerun on the DGX's 80 CPU cores (24 workers; the laptop run was reaped for low memory). Init from the camp-trained
net (r5: 3% extracted, 20% alive), teacher `lookahead:60:4` (camp_v2 before t60). Round 0 clones the teacher's own
rollouts: 17,103 states, train acc 93.9%, eval (200 seeds) **1.5% extracted, 8.5% alive, life 73.3** - worse than
its camp-trained start, as expected when cloning only teacher-visited states (DAgger rounds 1-5 next). 917 s/round.
Laya r1 (DAgger vs camp): 13.0% extracted / 33.5% alive / life 82.3, flat vs r0 13.5%; r2 fine-tune running on GPU 2.

---

## 2026-10-09 14:30 — Tiny CNN fails to learn from the lookahead teacher (manager)

Rerun after the worker fix (20d826c), DGX CPUs, ~4-5 min/round, init from the camp-trained net. 200 eval seeds:

| round | states | train acc | A0 extracted | A0 alive | life | hunger deaths |
|---|---|---|---|---|---|---|
| 0 | 17,103 | 93.9% | 1.5% | 8.5% | 73.3 | 35 |
| 1 | 31,364 | 98.6% | 0% | 6.5% | 72.5 | 36 |
| 2 | 45,709 | 98.9% | 0.5% | 8% | 73.8 | 30 |
| 3 | 60,249 | 99.1% | 0% | 5.5% | 73.5 | 37 |
| 4 | 75,324 | 99.1% | 0% | 8.5% | 76.0 | 36 |
| 5 | 90,366 | 99.0% | 0.5% | 8% | 75.7 | 44 |

Worse than the same net taught by camp (r5: 3% / 20% alive / 78.3) and flat across DAgger. Train accuracy 99% with
no eval gain = it memorises labels that don't form a learnable policy. Likely: the lookahead's choice depends on
4 noisy rollouts (similar states get different labels from call to call), and its advantage is reacting to exact
zombie positions by simulation, which a reactive net can't reproduce from one frame. Hunger deaths roughly tripled
(camp_v2 continuation eats badly; the net copies that). Not a dead end for the teacher: next try (proposal) is
denoising labels (M=16, or label with the lookahead's action *distribution* as a soft target) before spending it
on Laya. Data: `data/2026-10-09_tiny_lookahead.jsonl`.

---

## 2026-10-09 15:30 — Laya r2: 11.5% extracted, 38% alive (DAgger vs camp plateau) (manager)

200 seeds: r0 13.5% / 34.5% alive, r1 13.0% / 33.5%, **r2 11.5% / 38.0% / life 82.5** (thirst 87, zombie 20).
Within noise (SE ~2.4 pts) of each other and just under camp (17.5% / ~37%). Survival matches the teacher; the gap
is the helicopter run. r3 started 15:18 on GPU 2.

---

## 2026-10-09 18:20 — Final: three students vs camp on v3-rc2 (manager)

200 seeds `Random(999)`, A0 healthy, teammates camp. Teacher for all students: camp.

| A0 policy | params | training | A0 extracted | A0 alive | A0 life | ms/decision |
|---|---|---|---|---|---|---|
| camp (teacher) | - | rules | 17.5% | ~37% | 82.0 | <1 |
| **Laya r0** (cloning) | 421M | ~2 h V100 | **13.5%** | 34.5% | 82.2 | ~71 |
| Laya r1 (DAgger) | 421M | | 13.0% | 33.5% | 82.3 | 68 |
| Laya r2 | 421M | | 11.5% | **38.0%** | 82.5 | |
| Laya r3 | 421M | ~11 h total | 12.5% | 30.0% | 80.5 | 53 |
| tiny CNN r5 (DAgger) | 552k | ~1 h laptop CPU | 3% | 20% | 78.3 | ~1-2 |
| Qwen 3B K=3 r5 (DAgger)* | 3B | ~25 h V100 | 0% | 0-7% | 77.5 | ~1000 |
| Laya zero-shot (30 eps) | 421M | none | 0% | 0% | 24.0 | 59 |

*Qwen numbers are its own eval (60 seeds 4321; camp on those seeds: 13%).

Laya plateaus at 12-13% from the first round; DAgger against camp adds nothing beyond cloning (r0-r3 within the
~2.4 pt SE). It matches camp on survival but not on the helicopter run. The lookahead teacher (30%) is the only
route above camp found so far, but its raw labels did not transfer to the tiny CNN (see 14:30); denoise them
before training Laya on them.

---

## 2026-10-10 01:50 — Lookahead-improved labels: great teacher, worse Laya (manager)

**Labels** (`tools/probes/lookahead_labels.py`, DGX CPUs, 64 workers, 29 min): 600 episodes driven by camp_v2 with
the lookahead overriding only when its best action beats camp_v2's own by > 0.05 (M=16 fair rollouts, from t55),
5% random actions. 45,194 rows, 3,169 overrides (7%). The driving policy itself scored **31.8% A0 extracted, 43.8%
alive** - the best public-information policy so far (camp 17.5% / ~37%). (A first try at 20% random actions killed
A0 before t55 in most episodes; restarted at 5%.) Raw soft labels were abandoned: before ~t60 every action's value
ties at ~0.1, so argmax/softmax labels there are rollout noise - the likely reason the tiny CNN learned nothing.

**Laya** continued from `laya_r3` on these labels (waits subsampled, 2 epochs, lr 1.5e-5/6e-5): **6.0% A0
extracted, 18% alive, life 70.2, hunger 91/200** (r3: 12.5% / 30% / 80.5, hunger 10). A regression, dominated by
starvation. Not diagnosed yet. Suspects: (1) the new set has no DAgger rows from Laya's own states, so the
continued fine-tune overwrote what r1-r3 taught about recovering from its own mistakes; (2) the eat decision: the
override that most often fires is eat-vs-pickup, and camp_v2's eat timing plus subsampled waits may have shifted
eating late. Next: mix these labels with the r0-r3 aggregate instead of replacing it, and check eat timing in
replays. Data: `data/2026-10-09_laya_dagger.jsonl` (last row).

---

## 2026-10-10 04:30 — Laya mix stage A: starvation fixed, back to r3 level (manager)

Diagnosis of the 01:50 regression: camp eats at hunger ~41 (inside the safehouse healing cancels starvation, so it
eats right before leaving), the improved teacher eats at ~15; trained on the new labels alone Laya learned neither
and starved on the run. Stage A of `tools/laya_mix.sh`: fine-tune from `laya_r3` for 1 epoch on the r0-r3 camp
aggregate (37,279 rows after wait-capping) + the improved-teacher labels (39,468).
200 seeds: **11.5% A0 extracted, 30% alive, life 81.3, hunger 18** (was 6% / 18% / 70.2 / hunger 91; r3 12.5%).
Stage B (Laya plays 150 recorded episodes, seeds 6161) scored 20% extracted / 42% alive on its own seeds; the
gap to the 200-seed eval is mostly seed-set variance (SE ~3 pts each). Stage C (CPU replay labelling) running.
