"""Composable reward rubrics for SurviveCity v2.

Every rubric is a pure function of (EpisodeState, agent_id). compose_reward
sums them and clips to (0.01, 0.99) for OpenEnv compliance. Any per-step
flag the rubrics depend on (ate_this_step, drank_this_step, ...) is reset
by game.advance_step before the next round.

Rubrics:
  v1-derived:
    1. survival          — dense, per-step
    2. iterated_vote     — fires once at each of t=30, 50, 70, 90; per-vote
    3. group_outcome     — terminal
  v2-new (original 7):
    4. thirst            — dense, per-step
    5. broadcast_economy — per-broadcast-over-threshold
    6. night_survival    — dense in night windows
    7. infection_dodge   — dense + one-shot on latent->revealed transition
    8. medication        — one-shot on inject outcome
    9. hoarding_penalty  — terminal, per unused inventory slot
   10. wave_survival     — one-shot per wave step
  v2.1 — credit-assignment & shaping fixes (this revision):
   11. forage_shaping    — dense potential toward nearest food/water when hungry/thirsty
   12. zombie_proximity  — dense penalty proportional to 1/dist_to_nearest_zombie when close
   13. anti_camp         — small per-step nudge against unbroken wait-streaks while at risk
   14. infected_deception — terminal bonus for starting infected who stayed hidden / framed others

The v2.1 rubrics were added because v1 eval transcripts showed agents
starving in place: there was no gradient toward food/water, no penalty for
sitting next to a zombie, and no incentive for the infected role to play
its asymmetry. Each new rubric has small magnitudes (≤0.05/step) so the
existing terminal/vote signals still dominate end-of-episode credit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from survivecity_v2_env.spawn import WAVE_SCHEDULE
from survivecity_v2_env.layout import (
    FOOD_CELLS, WATER_CELLS, MEDICINE_CELLS,
)

if TYPE_CHECKING:
    from survivecity_v2_env.game import EpisodeState


# OpenEnv strict bounds: rewards must be in the OPEN interval (0.0, 1.0).
_SCORE_MIN = 0.01
_SCORE_MAX = 0.99


def _clip(score: float) -> float:
    return max(_SCORE_MIN, min(_SCORE_MAX, score))


# Vote phases by step. game.py resolves them in advance_step at step T+1.
# Schedule rationale: 30 (5 steps post-biter-reveal), 50 (mid-game retry),
# 70 (10 steps post-saboteur-reveal so cues accumulate), 90 (final, lockout
# still has 9 steps to bite before episode end at 100).
VOTE_PHASES: list[int] = [30, 50, 70, 90]


# ---------------------------------------------------------------------------
# 1. survival_reward (per-step, per-agent) — same shape as v1
# ---------------------------------------------------------------------------

def survival_reward(state: "EpisodeState", agent_id: int) -> float:
    a = state.agents[agent_id]
    r = 0.0
    if a.is_alive:
        r += 0.005
        if a.ate_this_step:
            r += 0.05
        if a.hunger >= 10:
            r -= 0.05
        if a.damage_this_step > 0:
            r -= 0.10 * a.damage_this_step
    if a.died_this_step:
        r -= 0.50
    return r


# ---------------------------------------------------------------------------
# 2. iterated_vote_reward — fires for each resolved vote phase
# ---------------------------------------------------------------------------

def iterated_vote_reward(state: "EpisodeState", agent_id: int) -> float:
    """Sum the reward contribution from each *just-resolved* vote phase.

    `state.vote_phase_just_resolved` is set by game.advance_step in the step
    immediately after the vote step (T+1). It is a dict {phase_step: True}
    listing phases resolved during the previous step. We score each.
    """
    if not state.vote_phase_just_resolved:
        return 0.0

    a = state.agents[agent_id]
    total = 0.0
    is_starting_infected = a.infection_role in {"biter", "saboteur"}

    for phase_step in list(state.vote_phase_just_resolved.keys()):
        votes = state.votes_cast.get(phase_step, {})
        target = votes.get(agent_id)
        # The phase produced a lockout iff phase_step in state.lockout_results
        lockout_target = state.lockout_results.get(phase_step)

        # Per-voter contribution
        if is_starting_infected:
            # The infected wants to frame healthy agents.
            if target is None:
                total += -0.05
            elif target == agent_id:
                total += -0.30
            else:
                # Voted against another agent — infected celebrates IF that
                # other agent was healthy AND was actually locked out.
                target_a = state.agents[target]
                target_was_healthy = target_a.infection_role is None
                if lockout_target == target and target_was_healthy:
                    total += 0.30
                else:
                    total += 0.10  # contributed to misdirection even if no lockout
        else:
            if target is None:
                total += -0.05
            elif state.agents[target].infection_role in {"biter", "saboteur"}:
                # Voted against a true (starting) infected — always rewarded
                total += 0.30
                if lockout_target == target:
                    total += 0.10  # actually got them locked out
            else:
                # Voted against a healthy agent
                total += -0.20

    return total


# ---------------------------------------------------------------------------
# 3. group_outcome_reward — terminal
# ---------------------------------------------------------------------------

def group_outcome_reward(state: "EpisodeState", agent_id: int) -> float:
    if not state.done:
        return 0.0

    a = state.agents[agent_id]
    is_starting_infected = a.infection_role in {"biter", "saboteur"}

    healthy_alive = sum(
        1 for ag in state.agents
        if ag.is_alive and ag.infection_state == "none"
    )
    starting_infected_neutralised = all(
        (not ag.is_alive) or ag.locked_out
        for ag in state.agents
        if ag.infection_role in {"biter", "saboteur"}
    )

    r = 0.0
    if healthy_alive >= 1:
        # Group survived
        if a.is_alive and a.infection_state == "none":
            r += 0.40
        if starting_infected_neutralised and not is_starting_infected:
            r += 0.30
    else:
        # Healthy team wiped
        if is_starting_infected and a.is_alive:
            r += 0.40
        elif not a.is_alive and not is_starting_infected:
            r -= 0.20
    return r


# ---------------------------------------------------------------------------
# 4. thirst_reward — dense, per-step
# ---------------------------------------------------------------------------

def thirst_reward(state: "EpisodeState", agent_id: int) -> float:
    a = state.agents[agent_id]
    if not a.is_alive:
        return 0.0
    if a.drank_this_step:
        return 0.03
    if a.thirst >= 10:
        return -0.05
    return 0.005


# ---------------------------------------------------------------------------
# 5. broadcast_economy_reward — penalise loud broadcasts
# ---------------------------------------------------------------------------

def broadcast_economy_reward(state: "EpisodeState", agent_id: int) -> float:
    if state.broadcasts_over_threshold_this_step.get(agent_id, 0) > 0:
        return -0.02 * state.broadcasts_over_threshold_this_step[agent_id]
    return 0.0


# ---------------------------------------------------------------------------
# 6. night_survival_reward — dense bonus during night windows
# ---------------------------------------------------------------------------

def night_survival_reward(state: "EpisodeState", agent_id: int) -> float:
    a = state.agents[agent_id]
    if not a.is_alive:
        return 0.0
    # Night windows: 25-49 and 75-99
    s = state.step_count
    if 25 <= s <= 49 or 75 <= s <= 99:
        return 0.01
    return 0.0


# ---------------------------------------------------------------------------
# 7. infection_dodge_reward — dense + transition penalty
# ---------------------------------------------------------------------------

def infection_dodge_reward(state: "EpisodeState", agent_id: int) -> float:
    a = state.agents[agent_id]
    r = 0.0
    if a.is_alive and a.infection_state == "none":
        r += 0.02
    # One-shot penalty when latent->revealed transitioned this step
    if state.latent_revealed_this_step.get(agent_id, False):
        r -= 0.10
    return r


# ---------------------------------------------------------------------------
# 8. medication_reward — one-shot on inject outcome
# ---------------------------------------------------------------------------

def medication_reward(state: "EpisodeState", agent_id: int) -> float:
    """Score the inject action's outcome this step (if any)."""
    rec = state.last_inject_result.get(agent_id)
    if rec is None:
        return 0.0
    if rec == "self_cured":
        return 0.30
    if rec == "other_cured":
        return 0.40
    if rec == "wasted_on_healthy":
        return -0.05
    if rec == "wasted_on_revealed":
        return -0.05
    if rec == "no_inventory":
        return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# 9. hoarding_penalty_reward — terminal, per unused inventory slot
# ---------------------------------------------------------------------------

def hoarding_penalty_reward(state: "EpisodeState", agent_id: int) -> float:
    if not state.done:
        return 0.0
    a = state.agents[agent_id]
    return -0.05 * len(a.inventory)


# ---------------------------------------------------------------------------
# 10. wave_survival_reward — one-shot per wave step
# ---------------------------------------------------------------------------

def wave_survival_reward(state: "EpisodeState", agent_id: int) -> float:
    a = state.agents[agent_id]
    s = state.step_count
    # We pay this on the step IMMEDIATELY AFTER the wave so the agent had a
    # chance to be killed in the wave-step itself.
    payout_steps = {ws + 1 for ws in WAVE_SCHEDULE}
    if s in payout_steps and a.is_alive:
        return 0.05
    return 0.0


# ---------------------------------------------------------------------------
# 11. forage_shaping_reward — dense potential-based shaping toward food/water
# ---------------------------------------------------------------------------

def _manhattan(a_row: int, a_col: int, cells) -> int:
    """Min Manhattan distance from (a_row,a_col) to any cell in `cells`.

    Returns 999 if `cells` is empty so the caller can shortcut.
    """
    if not cells:
        return 999
    best = 999
    for (r, c) in cells:
        d = abs(a_row - r) + abs(a_col - c)
        if d < best:
            best = d
    return best


def forage_shaping_reward(state: "EpisodeState", agent_id: int) -> float:
    """Dense gradient toward food when hungry, water when thirsty, medicine when latent.

    The eval transcripts showed ~75% of episodes ending in collective
    starvation by step 50: with 8 food cells on a 15×15 grid and a 100-step
    horizon, a random walk almost never reaches food before hunger kills
    the agent. There was nothing in the original 10 rubrics that nudged
    the policy toward food before the moment of eating. This rubric adds
    a tiny per-step potential that only kicks in when the agent is
    actually at risk (hunger>=7 or thirst>=7), so it doesn't bias play
    away from social-deduction once basic survival is solved.

    Magnitudes per step:
      * Hungry (hunger>=7): -0.003 * dist_to_nearest_food (clipped at 12)
      * Thirsty (thirst>=7): -0.003 * dist_to_nearest_water
      * Latent infected with medicine_used==0: -0.003 * dist_to_nearest_medicine
    """
    a = state.agents[agent_id]
    if not a.is_alive:
        return 0.0

    r = 0.0

    if a.hunger >= 7:
        # Live food cells only — depleted depots aren't reachable food yet
        live_food = [c for c in FOOD_CELLS if state.food_present.get(c, True)]
        d = _manhattan(a.row, a.col, live_food)
        r -= 0.003 * min(d, 12)

    if a.thirst >= 7:
        # Water depots are persistent (do not deplete) — always include all
        d = _manhattan(a.row, a.col, list(WATER_CELLS))
        r -= 0.003 * min(d, 12)

    # Latent infected without a cure path — nudge toward medicine cells
    if a.infection_state == "latent" and a.medicine_used == 0 and "medicine" not in a.inventory:
        live_med = [c for c in MEDICINE_CELLS if state.medicine_present.get(c, True)]
        d = _manhattan(a.row, a.col, live_med)
        r -= 0.003 * min(d, 12)

    return r


# ---------------------------------------------------------------------------
# 12. zombie_proximity_reward — dense penalty proportional to 1/dist when close
# ---------------------------------------------------------------------------

def zombie_proximity_reward(state: "EpisodeState", agent_id: int) -> float:
    """Penalise standing close to zombies. Encourages active avoidance.

    Without this, the only zombie-related signal is the -0.10 damage hit
    that fires on collision. By that point the agent has already lost HP;
    the gradient is too sparse to learn avoidance. We pay a small cost
    that scales as 1/distance for distance ∈ {1, 2, 3} so the agent feels
    "pressure" before the zombie reaches them.

    Capped at -0.03/step worst case (zombie 1 tile away). Safehouse cells
    are exempt — agents inside the safehouse are already protected by the
    no-zombie-can-enter rule, no extra incentive needed.
    """
    a = state.agents[agent_id]
    if not a.is_alive:
        return 0.0
    if (a.row, a.col) in {(r, c) for r in range(6, 9) for c in range(6, 9)}:
        return 0.0
    if not state.zombies:
        return 0.0
    best = 999
    for z in state.zombies:
        d = abs(a.row - z.row) + abs(a.col - z.col)
        if d < best:
            best = d
    if best >= 4:
        return 0.0
    if best == 0:
        # Already on the zombie — survival_reward already paid the damage.
        return -0.03
    return -0.03 / best


# ---------------------------------------------------------------------------
# 13. anti_camp_reward — small nudge against camping with hunger/thirst
# ---------------------------------------------------------------------------

def anti_camp_reward(state: "EpisodeState", agent_id: int) -> float:
    """Tiny penalty for sitting still while resources are running out.

    Works off `agent.waits_streak` which game.py increments on `wait` and
    resets on any movement/action. Once waits_streak >= 4 AND hunger or
    thirst has crossed the warning threshold, each additional wait costs
    -0.005. Caps the wait-spam exploit observed in v1 eval transcripts
    (agent stuck with hunger=12 spamming wait until death).
    """
    a = state.agents[agent_id]
    if not a.is_alive:
        return 0.0
    if a.waits_streak < 4:
        return 0.0
    if a.hunger < 8 and a.thirst < 8:
        return 0.0
    # Linear ramp: cap at -0.02 for very long streaks
    return -0.005 * min(a.waits_streak - 3, 4)


# ---------------------------------------------------------------------------
# 14b. scan_economy_reward — penalise consecutive scans (anti-exploit)
# ---------------------------------------------------------------------------

def scan_economy_reward(state: "EpisodeState", agent_id: int) -> float:
    """Penalise scan-spam to close the v1 exploit.

    During the 100-step run, `scan` accounted for 49.5% of all actions.
    It dodged every existing rubric guardrail: no movement (no
    zombie_proximity penalty), the legacy logic reset waits_streak (no
    anti_camp), no noise meter, only +1 thirst (silent until 7+).
    Combined with the parse_action substring bug, "no-op scan" became
    Pareto-better than `wait` for the policy. We close the loop by:

      1. Reclassifying invalid-target scans as `wait` in game.py so they
         can no longer dodge anti_camp.
      2. Penalising consecutive valid scans past the second here.

    Magnitude: -0.05 per scan beyond the second, capped at -0.30 (streak
    of 8+). Same order as a single bad vote; far smaller than terminal
    rubrics so it shapes behaviour without overwhelming social signals.
    """
    a = state.agents[agent_id]
    if not a.is_alive:
        return 0.0
    if a.scan_streak <= 2:
        return 0.0
    return -0.05 * min(a.scan_streak - 2, 6)


# ---------------------------------------------------------------------------
# 14. infected_deception_reward — terminal bonus for starting infected
# ---------------------------------------------------------------------------

def infected_deception_reward(state: "EpisodeState", agent_id: int) -> float:
    """Terminal-only bonus for the asymmetric infected role.

    Without this, biter/saboteur agents had the same per-step incentive as
    healthy agents (stay alive). The vote rubric is the only place
    asymmetry showed up, but it's an order of magnitude smaller than
    cumulative survival. Eval transcripts showed infected agents
    "winning" purely by accident (everyone else starved). This rubric
    pays a real terminal bonus for active deception:

      +0.30  if the starting infected reached step 100 still in `latent`
             (never revealed, hidden the whole game)
      +0.20  if at least one healthy agent got locked out by a vote that
             the infected voted for
      +0.10  per healthy agent killed by this biter (at most 4 healthy)

    Caps at +0.60 to keep the rubric within the same order of magnitude
    as group_outcome (+0.40 / +0.30).
    """
    if not state.done:
        return 0.0
    a = state.agents[agent_id]
    if a.infection_role not in {"biter", "saboteur"}:
        return 0.0

    r = 0.0
    if a.is_alive and a.infection_state == "latent":
        r += 0.30

    # Healthy locked out by a vote this agent participated in voting for
    framed = False
    for phase, target in state.lockout_results.items():
        if target is None:
            continue
        target_a = state.agents[target]
        if target_a.infection_role is None:  # was healthy
            voted_target = state.votes_cast.get(phase, {}).get(agent_id)
            if voted_target == target:
                framed = True
                break
    if framed:
        r += 0.20

    # Bites this agent landed (healthy victims they personally infected)
    bites_landed = sum(
        1 for b in state.bite_history
        if b.get("biter_id") == agent_id
    )
    r += 0.10 * min(bites_landed, 3)

    return min(r, 0.60)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

_RUBRIC_FUNCS = (
    survival_reward,
    iterated_vote_reward,
    group_outcome_reward,
    thirst_reward,
    broadcast_economy_reward,
    night_survival_reward,
    infection_dodge_reward,
    medication_reward,
    hoarding_penalty_reward,
    wave_survival_reward,
    forage_shaping_reward,
    zombie_proximity_reward,
    anti_camp_reward,
    scan_economy_reward,
    infected_deception_reward,
)

# Rubrics whose value is non-zero only when state.done. env.py settles
# these for ALL agents at the moment the episode terminates, so no agent
# loses terminal credit just because they weren't the last actor.
_TERMINAL_RUBRICS = (
    group_outcome_reward,
    hoarding_penalty_reward,
    infected_deception_reward,
)


def compose_reward(state: "EpisodeState", agent_id: int) -> tuple[float, float]:
    """Sum all rubrics and clip to (0.01, 0.99). Returns (clipped, raw)."""
    raw = sum(fn(state, agent_id) for fn in _RUBRIC_FUNCS)
    return _clip(raw), raw


def terminal_only_reward(state: "EpisodeState", agent_id: int) -> float:
    """Sum of just the terminal rubrics for end-of-episode settlement."""
    if not state.done:
        return 0.0
    return sum(fn(state, agent_id) for fn in _TERMINAL_RUBRICS)


def per_rubric_breakdown(state: "EpisodeState", agent_id: int) -> dict[str, float]:
    """Return each rubric's contribution this step. Used by the simulator."""
    return {fn.__name__: fn(state, agent_id) for fn in _RUBRIC_FUNCS}
