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
