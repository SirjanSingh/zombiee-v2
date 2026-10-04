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
