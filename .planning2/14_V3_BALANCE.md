# 14 — v3 balance (W3)

**Status:** done 2026-10-04. Default preset is now `v3-rc1` (`survivecity_v2_env/balance.py`).
`v2.2` stays available and is pinned by a golden trajectory hash and prompt-text hashes.

## Targets and result

100 episodes, seed 42, all 5 agents scripted by the same policy (`tools/calibrate.py`).
Survival = at least one healthy agent alive at t=100.

| policy | target | v2.2 | v3-rc1 |
|---|---|---|---|
| random | ~0% | 0% (ep 16.7) | 0% (ep 26.9) |
| heuristic_v2 (frozen, runs 1-4) | history only | 0% (ep 16.1) | 0% (ep 26.4) |
| heuristic_v3 (new baseline + GRPO rollout) | 5-20% | 0% (ep 26.2) | **11%** (ep 74.3) |
| camp planner | - | 0% (ep 66.8) | 73% (ep 99.0) |
| oracle | >= 60% | 0% (ep 66.8) | **83%** (ep 98.7) |

## Knobs (one line of reasoning each)

| knob | v2.2 | v3-rc1 | why |
|---|---|---|---|
| `hunger_rate`, `thirst_rate` | 1.0 | 0.6 | Planners died of thirst/hunger at ~t=67 because resupply after ~t=40 is suicidal. The clock rate sets the supply horizon: 0.75 still 0% for planners, 0.5 makes them ~98%, 0.6 puts them at 73-83%. Rates, not thresholds, so every "act at hunger 4/13/14" rule in policies stays meaningful. |
| `zombie_move_every` | 1 | 2 | Shamblers. A zombie-avoiding forager can only escape zombies that are slower than it; without this heuristic_v3 stays at 0-4%. |
| `zombie_chase_radius` | None | 4 | Zombies chased any agent anywhere outside the safehouse, so all of them converged on the safehouse door. Radius 4 lets short forage trips succeed while lingering near zombies is still deadly. Takes heuristic_v3 from 3% to 11-13%. |
| `starting_infected_progression` | True | False | Bug-like rule: the starting biter (bite step 0) died of infection_progression at t=30, 5 steps after its reveal, so bites were 0.00/episode and the t=30 vote was moot. Approved by the manager. Starting infected now live; bitten agents still die 30 steps after a bite without medicine. |

Not changed (single-knob sweeps showed little effect or they are social-deduction rules):
waves (halving or removing them moves the oracle by < 4 episode steps), food respawn delay,
`hp_max`, infected hunger multiplier, `p_bite`, reveal steps, vote steps.

## The heuristic was the bigger problem

`forage_heuristic_v2` survives 0% under every balance tried, even where planners reach 99%:
it drank forever once standing on water, "ate" on depleted depots, and walked into the walls next to
the inner-ring water. `forage_heuristic_v3` (`training/inference.py`) fixes those and adds a one-step
"don't step next to a zombie" rule; it is now `forage_heuristic_action`, i.e. the GRPO rollout policy
and the eval baseline. Runs 1-4 were scored against the broken v2 continuation.

## Synced with the config

- System prompt rules (clock, waves, zombie speed/radius, bitten-death rule, reveal steps, bite %),
  description (`Step t/max_steps`, `HP=x/hp_max`, latent reveal step) from the episode's config.
- Bite events are public in the description: `Bites seen: A2 bit A4 at t=31`.
- Rubric urgency thresholds (hungry at 10/15, urgency cap 12/15) are fractions of the config clocks.
- Postmortem latent reveal step uses `latent_duration`.
- Env metadata now includes `depleted_food` (public; the grid hides a depot under an agent).

Sweeps and raw data: `research_log/` (tags `v2.2-baseline`, `w3-combo1`, `v3-rc1`, `v3-rc1-final`,
`v2.2-heuristic_v3`), sweep grids in `tools/sweeps/`.
