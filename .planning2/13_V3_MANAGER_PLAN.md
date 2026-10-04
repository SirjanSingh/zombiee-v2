# 13 — v3 plan: make the env winnable, then train on the right states

**Status:** active. Manager session = `zombiee-v2-ce`, worker session = `zombiee-v2-9f`.
**Created:** 2026-10-04
**Task board:** `.planning2/TASKS.md` (single source of truth for who does what)

---

## Why runs 1-4 (and GiGPO run 6, if launched as-is) cannot work

Two independent blockers, both measured on CPU (60-100 episodes each, ~3 s per 60 eps):

### Blocker 1: the env is unwinnable as tuned

| policy (all 5 agents) | ep length | A0 lifetime | healthy alive at end | reached step 100 |
|---|---|---|---|---|
| `forage_heuristic_action` (current) | 16.0 | 13.0 | 0 | 0% |
| BFS planner, zombie-avoiding (`tools/probes/planner.py`) | 21-26 | 11-16 | 0 | 0% |
| water-camp planner (`tools/probes/planner2.py`) | 66.8 | 54.2 | 0 | 0% |
| **oracle** camp planner: knows who is infected, keeps distance, votes them out (`tools/probes/oracle.py`) | 66.8 | 49.0 | 0 | **0%** |

Even the oracle dies by ~step 67, mostly from thirst + hunger. Mechanics that cause it
(see `survivecity_v2_env/game.py`):

- Hunger/thirst +1 per turn (infected hunger 1.5x), -1 HP/turn each at >=15. HP max 3.
- Safehouse heals +1 HP/turn, so inside, hunger alone is net 0; hunger AND thirst is net -1.
- Zombies BFS-chase the nearest agent outside the safehouse and otherwise wander. With all
  agents inside they park on the perimeter, right on the inner food/water ring.
- Waves add 2/3/3 zombies at t=25/50/75 (up to 11). Resupply sorties after ~t=40 are suicide.
- Food depletes (10-step respawn). Water is infinite but outside.
- Revealed biter (t=25) bites adjacent agents at P=0.35/step, and a 3x3 safehouse makes
  everyone adjacent. Bitten agents die of infection_progression 30 steps later without medicine.
- Episode terminates when no healthy agent is alive, so "healthy alive at end" is 0 unless t=100.

### Blocker 2: training only ever sees the step-0 state

`training/train.py::build_scenario_dataset` builds every prompt from `env.reset(seed)`. All
agents always spawn at the same cells with hp=3, hunger=0, so the model trains on effectively
one situation and emits K actions open-loop. At eval it is queried on mid-game states it never
saw. GiGPO does not fix this: its anchors only cluster within the K open-loop steps after reset.

The current reward is actually a sound idea: model takes K actions, heuristic plays out the
rest, the return estimates Q^heuristic(s, a_1..K). Maximising that is **one step of policy
improvement over the heuristic**. It just has to happen at the states that matter.

---

## Plan

### Phase A: make the env winnable (CPU only, worker)
Target calibration on 100 episodes, all 5 agents scripted:
- current heuristic: survival (healthy >=1 at t=100) roughly 5-20%
- oracle / good planner: >= 60%
- random: ~0%
That gap is the headroom RL can climb. Changes must be config-driven (one dataclass of
balance knobs, defaults = new v3 values) with tests, and `training/inference.py` layout
copies kept in sync (see memory `hardcoded_layout_duplication`).

Candidate knobs (worker picks the smallest set that hits the targets, justify each):
starvation threshold / tick rate, zombie chase radius (wander unless agent within R),
wave sizes, safehouse size or a water source inside, food respawn delay, bite probability.

### Phase A2: a strategic objective, "rescue at turn 90" (user direction, 2026-10-04)
End goal: a model that actually survives the zombies. Make that the reward, not a side effect:
- **Rescue arrives at t=90.** Episode ends at t=90. Every agent alive, healthy, and inside the
  safehouse at t=90 is "rescued": large terminal reward (e.g. +5 per rescued agent to each
  healthy agent; trained agent gets its own + team share).
- **Milestones** at t=30 and t=60: small bonus (e.g. +0.5) per healthy agent still alive.
  These keep the signal from being purely sparse while staying aligned with the real goal.
- **Infected win** if they outnumber healthy agents at rescue; infected agents get the mirror
  reward. This keeps the social-deduction layer meaningful (vote out the infected, don't let
  a biter into the rescue).
- Demote the 15 shaping rubrics to a small weighted term (or drop) once the terminal signal
  works. Runs 1-3 were sunk by rubric gaming (scan-spam). Terminal survival is hard to game.
- Prompts tell the model the objective and the turns remaining ("Rescue in 37 turns").
- Calibration targets from Phase A apply to the rescue rate.

### Phase B: train on the states that matter (worker, after A)
1. Mid-episode start states: prompt = state after rolling the heuristic forward to a random
   t in [0, 90] from seed N. Embed `[SEED:N][T:t]`; reward fn replays deterministically to t.
2. Fixed-horizon return: score A0 over the next H steps (H ~20-30) with heuristic continuation,
   not until death. Lower variance, comparable across start times.
3. SFT warm-start from the best planner (the oracle minus oracle-only info), not the weak
   heuristic: build JSONL of (prompt, action) at A0 turns.

### Phase C: DGX runs (manager sets up infra, worker does code)
- DGX infra: repo on DGX is an untracked archive copy, no docker image, no Qwen in HF cache,
  Docker Hub blocked. Manager sets up a conda env + fresh clone.
- Run order: SFT warm-start -> GRPO (Phase B states) -> GiGPO (`--adv-estimator gigpo
  --prefix-actions 5 --gigpo-zscore`). Compare A0 lifetime and survival vs heuristic + planner.

### Metrics (headline, every eval)
A0 lifetime, mean healthy alive at t=100, survival rate, reached-t100 rate, mean reward,
action histogram. Never report survival alone (memory `eval_metric_binary_threshold`).
