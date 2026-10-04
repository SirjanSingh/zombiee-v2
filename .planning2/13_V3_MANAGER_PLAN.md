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

### Phase A2: a strategic objective, "radio + extraction" (manager design, 2026-10-04)
User's steer: the end goal is a model that really survives, with a strategic goal (the
"survive 90 turns" idea was an example, not a spec). Design chosen:
- **Radio at t=60** announces the extraction cell: one of the 4 corner regions, picked from the
  seed. Shown in every prompt from t=60 on ("Extraction at NE corner, helicopter lands t=85").
- **Helicopter at t=85-90.** Healthy agents standing in the extraction zone during the window
  are extracted (leave the map, safe). Big terminal reward per extracted healthy agent, shared
  with the team.
- **Contamination rule:** if any infected agent is in the zone when the helicopter leaves, the
  extraction fails for everyone (infected win). Deduction + voting now decide the outcome.
- Strategic arc the policy has to learn: early, stock water/food and avoid zombies; mid,
  identify and lock out the infected; late, time a cross-map run through zombies.
- Milestone bonus for healthy agents alive at t=30/60 (small) so the signal is not all at t=90.
- Shaping rubrics demoted to a small weighted term once the terminal signal works (runs 1-3 were
  sunk by rubric gaming; extraction is hard to game).
- Calibration: heuristic extraction rate ~5-20%, oracle planner >= 60%.

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

---

## Research update (2026-10-04, manager; abstracts read, papers not yet read in full)

What changed since plan 11 (May-June 2026) and how it applies here:

- **Signal dilution** ([Drowning in Routine, arXiv 2606.22164](https://arxiv.org/abs/2606.22164), ICML'26 FAGEN workshop):
  routine turns add gradient variance to trajectory-level estimators like GRPO without adding
  signal; turn-level vs trajectory-level SNR scales as rho^(-1/2) with decision density rho.
  Our game is low-density (many "wait in safehouse" turns). **Apply:** in Phase B, sample start
  states weighted toward decision-dense moments (low water/food, zombie within 3, vote turns,
  radio/extraction window) instead of uniform t. Supports turn-level credit (GiGPO) over plain GRPO.
- **ReBN** ([GEM, ICLR'26, arXiv 2510.01051](https://arxiv.org/abs/2510.01051)): REINFORCE with return
  batch normalization handles dense per-turn rewards with better credit assignment than GRPO in
  multi-turn settings. **Apply:** a cheap alternative/baseline to GiGPO once per-turn rewards exist;
  worth an ablation (GRPO vs GiGPO vs ReBN) since the video story benefits from a comparison.
- **State2State** ([arXiv 2608.04934](https://arxiv.org/abs/2608.04934), work in progress): mid-training on
  tasks derived from explored environment states ("reach target state"), verified by rule-based state
  matching; helps as standalone and as RL init on ALFWorld/ScienceWorld. **Apply:** confirms the
  direction of training from explored mid-episode states rather than resets.
- Also seen, not yet assessed: [Tree-GRPO (ICLR'26)](https://github.com/AMAP-ML/Tree-GRPO) (tree-search rollouts,
  Qwen2.5-3B, 1/4 rollout budget), [Granularity-adaptive credit assignment (2609.12424)](https://arxiv.org/pdf/2609.12424),
  [TRACE turn-level credit (2607.13988)](https://arxiv.org/pdf/2607.13988),
  [PGPO potential-guided (2609.02236)](https://arxiv.org/abs/2609.02236).

Net: algorithm choice is NOT the bottleneck right now (env winnability and state distribution are).
Keep GiGPO as the main method, add ReBN as a comparison arm after Phase B works.
