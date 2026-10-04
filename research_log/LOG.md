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
