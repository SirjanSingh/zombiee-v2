"""Tests for SurviveCity v2 mechanics.

Run with:
    cd v2 && pytest -q
"""

from __future__ import annotations

import os
import random

import pytest

from survivecity_v2_env.env import SurviveCityV2Env
from survivecity_v2_env.game import (
    create_episode,
    apply_agent_action,
    advance_step,
    advance_zombies,
)
from survivecity_v2_env.infection import should_bite, _hash01, P_BITE
from survivecity_v2_env import inventory as inv
from survivecity_v2_env import spawn as spawn_mod


# ---------------------------------------------------------------------------
# Reset / step basics
# ---------------------------------------------------------------------------

def test_reset_basics():
    env = SurviveCityV2Env()
    obs = env.reset(seed=1)
    assert len(obs["agents"]) == 5
    assert len(obs["zombies"]) == 3
    assert obs["step_count"] == 0
    # Two starting infected (one biter, one saboteur)
    starting = obs["metadata"]["starting_infected"]
    assert len(starting) == 2
    assert obs["metadata"]["phase"] in {"pre_biter_reveal", "post_biter_reveal", "mid_episode", "post_saboteur_reveal"}


def test_reward_in_open_unit_interval():
    """OpenEnv contract: every observed reward must be strictly in (0, 1)."""
    env = SurviveCityV2Env()
    obs = env.reset(seed=2)
    rng = random.Random(2)
    actions = ["move_up", "move_down", "eat", "wait", "drink", "pickup"]
    for _ in range(100):
        if obs["done"]:
            break
        aid = obs["metadata"]["current_agent_id"]
        a = {"agent_id": aid, "action_type": rng.choice(actions)}
        obs = env.step(a)
        r = obs["reward"]
        assert 0.0 < r < 1.0, f"reward {r} not in (0,1)"


def test_action_space_includes_v1_and_v2():
    """All v1 + v2 action types must validate without errors."""
    env = SurviveCityV2Env()
    obs = env.reset(seed=3)
    every_action = [
        "move_up", "move_down", "move_left", "move_right",
        "eat", "wait", "vote_lockout", "broadcast",
        "drink", "scan", "pickup", "drop", "give", "inject",
    ]
    for atype in every_action:
        if obs["done"]:
            break
        aid = obs["metadata"]["current_agent_id"]
        action = {
            "agent_id": aid,
            "action_type": atype,
            "vote_target": 1,
            "message": "hi",
            "scan_target": 2,
            "inject_target": 0,
            "gift_target": 1,
            "item_slot": 0,
            "item_type": "food",
        }
        obs = env.step(action)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def _run_random_episode(seed: int, n_actions: int = 50) -> list[tuple]:
    env = SurviveCityV2Env()
    obs = env.reset(seed=seed)
    rng = random.Random(seed)
    history = []
    actions_pool = ["move_up", "move_down", "move_left", "move_right", "eat", "drink", "wait"]
    for _ in range(n_actions):
        if obs["done"]:
            break
        aid = obs["metadata"]["current_agent_id"]
        action = {"agent_id": aid, "action_type": rng.choice(actions_pool)}
        obs = env.step(action)
        history.append((
            obs["step_count"],
            obs["reward"],
            obs["metadata"]["n_alive"],
            obs["metadata"]["n_zombies"],
            len(obs["metadata"]["bite_history"]),
        ))
    return history


def test_episode_determinism():
    """Same seed should produce identical state trajectories."""
    h1 = _run_random_episode(42)
    h2 = _run_random_episode(42)
    assert h1 == h2


# ---------------------------------------------------------------------------
# Bite RNG
# ---------------------------------------------------------------------------

def test_bite_rng_is_deterministic():
    a = should_bite(123, 25, 0, 1)
    b = should_bite(123, 25, 0, 1)
    assert a == b


def test_bite_rng_distribution_matches_p_bite():
    """Empirical bite frequency over 5000 samples should be close to P_BITE."""
    samples = [should_bite(i, 25, 0, 1) for i in range(5000)]
    freq = sum(samples) / len(samples)
    assert abs(freq - P_BITE) < 0.03, f"bite freq {freq} != {P_BITE} (±0.03)"


def test_hash01_is_uniform():
    samples = [_hash01(i) for i in range(5000)]
    mean = sum(samples) / len(samples)
    assert 0.45 < mean < 0.55


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

def test_inventory_cap():
    items: list[str] = []
    assert inv.add_item(items, "food")
    assert inv.add_item(items, "water")
    assert inv.add_item(items, "medicine")
    assert not inv.add_item(items, "food")  # full
    assert len(items) == 3


def test_inventory_remove_at():
    items = ["food", "water", "medicine"]
    assert inv.remove_at(items, 1) == "water"
    assert items == ["food", "medicine"]
    # Out of range
    assert inv.remove_at(items, 99) is None
    assert inv.remove_at(items, None) is None


def test_inventory_remove_first():
    items = ["food", "food", "medicine"]
    assert inv.remove_first(items, "food")
    assert items == ["food", "medicine"]
    assert not inv.remove_first(items, "water")  # no water


# ---------------------------------------------------------------------------
# Wave spawning
# ---------------------------------------------------------------------------

def test_wave_step_lookup():
    assert spawn_mod.is_wave_step(25)
    assert spawn_mod.is_wave_step(50)
    assert spawn_mod.is_wave_step(75)
    assert not spawn_mod.is_wave_step(0)
    assert not spawn_mod.is_wave_step(99)


def test_wave_spawn_count_capped():
    rng = random.Random(0)
    # If already at the cap, pick_wave_spawn_cells returns []
    cells = spawn_mod.pick_wave_spawn_cells(50, rng, occupied=[], current_zombie_count=12)
    assert cells == []


def test_wave_spawn_excludes_occupied():
    rng = random.Random(0)
    pool_size = len(spawn_mod.WAVE_SPAWN_POOL)
    # Occupy every pool cell -> no spawn possible
    cells = spawn_mod.pick_wave_spawn_cells(50, rng, occupied=spawn_mod.WAVE_SPAWN_POOL, current_zombie_count=0)
    assert cells == []
    # Otherwise we should get up to WAVE_SCHEDULE[50] cells
    rng2 = random.Random(0)
    cells2 = spawn_mod.pick_wave_spawn_cells(50, rng2, occupied=[], current_zombie_count=0)
    assert len(cells2) == spawn_mod.WAVE_SCHEDULE[50]
    # All chosen cells are inside the pool
    for c in cells2:
        assert c in spawn_mod.WAVE_SPAWN_POOL


# ---------------------------------------------------------------------------
# Vote resolution (manually driven)
# ---------------------------------------------------------------------------

def test_vote_resolution_manual():
    ep = create_episode(seed=0)
    ep.step_count = 30
    for aid in range(5):
        apply_agent_action(ep, aid, "vote_lockout", vote_target=4)
    advance_step(ep)
    assert ep.lockout_results.get(30) == 4
    assert ep.agents[4].locked_out
    assert 30 in ep.vote_resolved_steps


def test_vote_no_plurality():
    ep = create_episode(seed=0)
    ep.step_count = 30
    # 2 vote for A1, 2 vote for A2, 1 abstain
    apply_agent_action(ep, 0, "vote_lockout", vote_target=1)
    apply_agent_action(ep, 1, "vote_lockout", vote_target=1)
    apply_agent_action(ep, 2, "vote_lockout", vote_target=2)
    apply_agent_action(ep, 3, "vote_lockout", vote_target=2)
    apply_agent_action(ep, 4, "vote_lockout", vote_target=None)
    advance_step(ep)
    assert ep.lockout_results.get(30) is None
    assert not any(a.locked_out for a in ep.agents)


# ---------------------------------------------------------------------------
# Inject / medicine
# ---------------------------------------------------------------------------

def test_inject_self_cures_latent():
    ep = create_episode(seed=0)
    # Force agent 1 to be latent-bitten
    ep.agents[1].infection_state = "latent"
    ep.agents[1].infection_role = "biter"
    ep.agents[1].bite_at_step = 5
    ep.agents[1].inventory = ["medicine"]
    apply_agent_action(ep, 1, "inject", inject_target=1, item_slot=0)
    assert ep.agents[1].infection_state == "none"
    assert ep.agents[1].inventory == []
    assert ep.last_inject_result.get(1) == "self_cured"


def test_inject_wasted_on_healthy():
    ep = create_episode(seed=0)
    ep.agents[2].inventory = ["medicine"]
    # Agent 2 is healthy by default (unless they were one of the two starting
    # infected). Force to none for determinism.
    ep.agents[2].infection_state = "none"
    ep.agents[2].infection_role = None
    apply_agent_action(ep, 2, "inject", inject_target=2, item_slot=0)
    assert ep.agents[2].inventory == []
    assert ep.last_inject_result.get(2) == "wasted_on_healthy"


# ---------------------------------------------------------------------------
# Day/night
# ---------------------------------------------------------------------------

def test_day_night_phase_advancement():
    ep = create_episode(seed=0)
    ep.step_count = -1  # so first advance_step lands on 0
    advance_step(ep)
    assert ep.day_phase == "day"
    ep.step_count = 24
    advance_step(ep)
    assert ep.day_phase == "night"
    ep.step_count = 49
    advance_step(ep)
    assert ep.day_phase == "day"
    ep.step_count = 74
    advance_step(ep)
    assert ep.day_phase == "night"


# ---------------------------------------------------------------------------
# v2.1 — pending_reward credit assignment
# ---------------------------------------------------------------------------

def test_zombie_damage_credited_via_pending_reward():
    """Zombie damage to a non-acting agent must reach cumulative_rewards.

    Repro of the v1 bug: in v1, damage taken while another agent was acting
    (or during the zombie phase) was placed on damage_this_step and then
    wiped on the victim's next reset_step_flags. v2.1 routes such damage
    through pending_reward so it survives into cumulative_rewards.
    """
    ep = create_episode(seed=0)
    # Force agent 2 onto a zombie's cell so the next advance_zombies hits.
    ep.agents[2].row = ep.zombies[0].row
    ep.agents[2].col = ep.zombies[0].col
    initial_pending = ep.agents[2].pending_reward
    advance_zombies(ep)
    # The zombie phase deposits the damage cost into pending_reward, NOT
    # damage_this_step (which would be wiped before the victim acted).
    assert ep.agents[2].pending_reward < initial_pending
    # And no out-of-band damage_this_step was set on the victim.
    assert ep.agents[2].damage_this_step == 0


def test_pending_reward_drained_into_cumulative():
    """When the victim acts next, pending_reward is drained into cumulative."""
    env = SurviveCityV2Env()
    env.reset(seed=11)
    ep = env._episode
    # Park agent 0 (the next actor) on a zombie tile so advance_zombies hits.
    # We need to step through agents 0..4 first; easier: set up a one-step
    # fake by stuffing pending_reward directly.
    ep.agents[0].pending_reward = -0.40
    obs = env.step({"agent_id": 0, "action_type": "wait"})
    assert env._cumulative_rewards[0] <= -0.30  # pending got drained in
    assert ep.agents[0].pending_reward == 0.0


def test_terminal_settlement_pays_all_agents():
    """When the episode terminates, every agent should see terminal credit.

    Without the env.py terminal-settle pass, only the agent who acted last
    on the terminal step would see group_outcome / hoarding_penalty /
    infected_deception. Verify all alive starting-infected agents see
    terminal credit.
    """
    env = SurviveCityV2Env()
    env.reset(seed=42)
    ep = env._episode
    # Force terminal: kill all healthy agents -> infected wins
    healthy = [a for a in ep.agents if a.infection_role is None]
    for a in healthy:
        a.is_alive = False
        a.hp = 0
    # Step through every alive agent so the round closes and advance_step
    # detects the terminal condition. With 2 infected alive we need 2
    # actions to close the round.
    obs = None
    for _ in range(5):
        if obs is not None and obs["done"]:
            break
        next_id = env._get_next_alive_agent()
        if next_id is None:
            break
        obs = env.step({"agent_id": next_id, "action_type": "wait"})
    assert obs is not None and obs["done"]
    # Every alive starting-infected agent should have non-zero cumulative
    # (group_outcome +0.40 for them since healthy_alive==0)
    starting_infected = [a.agent_id for a in ep.agents if a.infection_role in {"biter", "saboteur"}]
    for sid in starting_infected:
        if ep.agents[sid].is_alive:
            assert env._cumulative_rewards[sid] > 0, f"infected A{sid} got no terminal credit"


# ---------------------------------------------------------------------------
# v2.1 — shaping rubrics
# ---------------------------------------------------------------------------

def test_forage_shaping_rewards_proximity_to_food():
    """Hungry agent close to food should have higher forage_shaping than far."""
    from survivecity_v2_env.rubric import forage_shaping_reward
    ep = create_episode(seed=0)
    ep.agents[0].hunger = 8
    # Place near food cell (1,1)
    ep.agents[0].row, ep.agents[0].col = 2, 2
    near = forage_shaping_reward(ep, 0)
    # Place far from any food
    ep.agents[0].row, ep.agents[0].col = 7, 7
    far = forage_shaping_reward(ep, 0)
    assert near > far  # closer = less negative penalty


def test_forage_shaping_off_when_not_hungry():
    """Forage shaping is zero only when hunger AND thirst are 0 (latent agents always-on)."""
    from survivecity_v2_env.rubric import forage_shaping_reward
    ep = create_episode(seed=0)
    # Use a known-healthy agent (one of the three NOT in starting_infected)
    starters = {a.agent_id for a in ep.agents if a.infection_role}
    healthy_id = next(i for i in range(5) if i not in starters)
    ep.agents[healthy_id].hunger = 0
    ep.agents[healthy_id].thirst = 0
    assert forage_shaping_reward(ep, healthy_id) == 0.0


def test_forage_shaping_scales_with_hunger():
    """Magnitude scales linearly with hunger so the model has gradient even early.

    Run-2 history: original threshold>=7 left a dead zone for hunger 1-6,
    so model couldn't tell "move toward food" from "stay put" at step 0.
    Now fires at hunger>=1 with magnitude scaled by hunger/12.
    """
    from survivecity_v2_env.rubric import forage_shaping_reward
    ep = create_episode(seed=0)
    starters = {a.agent_id for a in ep.agents if a.infection_role}
    healthy_id = next(i for i in range(5) if i not in starters)
    ep.agents[healthy_id].thirst = 0
    # Pin position so distance is constant
    ep.agents[healthy_id].row, ep.agents[healthy_id].col = 7, 7

    ep.agents[healthy_id].hunger = 1
    weak = forage_shaping_reward(ep, healthy_id)
    ep.agents[healthy_id].hunger = 12
    strong = forage_shaping_reward(ep, healthy_id)

    assert weak < 0, f"hunger=1 should fire weak shaping, got {weak}"
    assert strong < weak, f"hunger=12 should be more negative than hunger=1: {strong} vs {weak}"
    # Strong should be ~12x weak (linear ramp 1/12 -> 12/12)
    assert abs(strong / weak - 12.0) < 0.5, f"scaling not ~12x: {strong/weak}"


def test_zombie_proximity_penalty_scales_with_distance():
    from survivecity_v2_env.rubric import zombie_proximity_reward
    ep = create_episode(seed=0)
    # Move agent 0 outside safehouse so the rubric applies
    ep.agents[0].row, ep.agents[0].col = 0, 0
    # Move zombie 0 next to agent 0 (distance 1)
    ep.zombies[0].row, ep.zombies[0].col = 0, 1
    close = zombie_proximity_reward(ep, 0)
    # Now move zombie far away — distance 5 → no penalty (rubric caps at <4)
    ep.zombies[0].row, ep.zombies[0].col = 14, 14
    for z in ep.zombies[1:]:
        z.row, z.col = 14, 14  # all zombies far
    far = zombie_proximity_reward(ep, 0)
    assert close < 0
    assert far == 0.0


def test_zombie_proximity_zero_in_safehouse():
    from survivecity_v2_env.rubric import zombie_proximity_reward
    ep = create_episode(seed=0)
    # All agents start in the safehouse — the rubric must return 0
    for i in range(5):
        assert zombie_proximity_reward(ep, i) == 0.0


def test_anti_camp_kicks_in_only_on_long_wait_streak_with_low_resources():
    from survivecity_v2_env.rubric import anti_camp_reward
    ep = create_episode(seed=0)
    a = ep.agents[0]
    # Short streak — no penalty
    a.waits_streak = 2
    a.hunger = 9
    assert anti_camp_reward(ep, 0) == 0.0
    # Long streak but resources fine — no penalty
    a.waits_streak = 6
    a.hunger = 2
    a.thirst = 2
    assert anti_camp_reward(ep, 0) == 0.0
    # Long streak AND low resources — penalty applies
    a.waits_streak = 6
    a.hunger = 9
    assert anti_camp_reward(ep, 0) < 0


def test_waits_streak_resets_on_non_wait_action():
    ep = create_episode(seed=0)
    a = ep.agents[0]
    a.waits_streak = 5
    apply_agent_action(ep, 0, "move_up")
    assert a.waits_streak == 0
    # Now wait should bring it back to 1
    apply_agent_action(ep, 0, "wait")
    assert a.waits_streak == 1


def test_infected_deception_terminal_pays_for_hidden_survival():
    from survivecity_v2_env.rubric import infected_deception_reward
    ep = create_episode(seed=0)
    # Pick the actual starting infected agent so the test is robust to seed
    inf_id = next(a.agent_id for a in ep.agents if a.infection_role)
    ep.agents[inf_id].infection_state = "latent"  # still hidden
    ep.agents[inf_id].is_alive = True
    ep.done = True
    r = infected_deception_reward(ep, inf_id)
    assert r >= 0.30  # at least the +0.30 hidden-survival bonus


def test_infected_deception_zero_for_healthy():
    from survivecity_v2_env.rubric import infected_deception_reward
    ep = create_episode(seed=0)
    starters = {a.agent_id for a in ep.agents if a.infection_role}
    healthy_id = next(i for i in range(5) if i not in starters)
    ep.done = True
    assert infected_deception_reward(ep, healthy_id) == 0.0


# ---------------------------------------------------------------------------
# v2.1 — observation metadata fields
# ---------------------------------------------------------------------------

def test_observation_exposes_last_actor_and_cumulative():
    """obs.metadata must distinguish current_agent (next to act) from last_actor."""
    env = SurviveCityV2Env()
    env.reset(seed=7)
    obs = env.step({"agent_id": 0, "action_type": "wait"})
    md = obs["metadata"]
    assert md["last_actor_id"] == 0
    assert md["current_agent_id"] != 0  # next agent should be 1 (or next alive)
    assert "cumulative_rewards" in md
    assert isinstance(md["cumulative_rewards"], dict)


# ---------------------------------------------------------------------------
# v2.1 — heuristic rollout policy
# ---------------------------------------------------------------------------

def test_forage_heuristic_targets_food_when_hungry():
    from training.inference import forage_heuristic_action
    obs = {
        "step_count": 5,
        "agents": [
            {"agent_id": 0, "row": 5, "col": 5, "hunger": 10, "thirst": 0, "is_alive": True},
        ],
    }
    a = forage_heuristic_action(0, obs)
    assert a["action_type"] in {"move_up", "move_down", "move_left", "move_right"}


def test_forage_heuristic_drinks_on_water_cell():
    from training.inference import forage_heuristic_action
    obs = {
        "step_count": 5,
        "agents": [
            # (3,3) is a water cell from layout.WATER_CELLS
            {"agent_id": 0, "row": 3, "col": 3, "hunger": 0, "thirst": 8, "is_alive": True},
        ],
    }
    a = forage_heuristic_action(0, obs)
    assert a["action_type"] == "drink"


def test_forage_heuristic_eats_on_food_cell():
    from training.inference import forage_heuristic_action
    obs = {
        "step_count": 5,
        "agents": [
            # (1,1) is a food cell
            {"agent_id": 0, "row": 1, "col": 1, "hunger": 8, "thirst": 0, "is_alive": True},
        ],
    }
    a = forage_heuristic_action(0, obs)
    assert a["action_type"] == "eat"


# ---------------------------------------------------------------------------
# v2.1 — metrics logger
# ---------------------------------------------------------------------------

def test_metrics_logger_writes_jsonl_and_summary(tmp_path):
    """MetricsLogger must produce one JSONL line per call and a valid summary."""
    from training.metrics import MetricsLogger
    metrics_path = str(tmp_path / "metrics.jsonl")
    ml = MetricsLogger(metrics_path)
    # Two synthetic reward_fn calls
    s1 = ml.log_reward_call(
        rewards=[-1.0, 0.5, -0.2, 0.3],
        cum_rewards=[-1.5, 0.0, -0.8, -0.4],
        rollout_lens=[15, 30, 22, 10],
        action_types=["move_up", "eat", "wait", "PARSE_FAIL"],
        parse_ok=3,
        rubric_breakdowns=[
            {"survival_reward": 0.005, "forage_shaping_reward": -0.01},
            {"survival_reward": 0.005, "forage_shaping_reward": -0.005},
        ],
        terminal_summary={
            "n_done": 4, "n_healthy_survived": 1, "n_infected_hidden": 0,
            "n_reached_vote30": 0, "n_reached_vote50": 0, "n_reached_vote70": 0, "n_reached_vote90": 0,
            "final_step_mean": 18.5,
        },
        training_step=1,
    )
    assert s1.n == 4
    assert s1.parse_ok == 3
    assert s1.r_mean == pytest.approx(-0.1, abs=1e-6)
    s2 = ml.log_reward_call(
        rewards=[0.1, 0.2, 0.3, 0.4],
        cum_rewards=[0.0, 0.0, 0.0, 0.0],
        rollout_lens=[40, 50, 60, 60],
        action_types=["eat", "eat", "drink", "drink"],
        parse_ok=4,
        rubric_breakdowns=None,
        terminal_summary=None,
        training_step=2,
    )
    assert s2.parse_ok == 4
    assert ml.call_count == 2

    # JSONL file should have exactly 2 valid rows
    import json
    with open(metrics_path) as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    assert len(lines) == 2
    rows = [json.loads(ln) for ln in lines]
    assert rows[0]["call_idx"] == 1
    assert rows[1]["call_idx"] == 2
    assert rows[0]["actions"] == {"move_up": 1, "eat": 1, "wait": 1, "PARSE_FAIL": 1}

    # Summary
    summary = ml.summary_dict()
    assert summary["n_calls"] == 2
    assert summary["r_mean_first"] == pytest.approx(-0.1, abs=1e-6)
    assert summary["actions_total"]["eat"] == 3  # 1 in s1 + 2 in s2
    summary_path = str(tmp_path / "summary.json")
    ml.write_summary_json(summary_path)
    with open(summary_path) as f:
        loaded = json.load(f)
    assert loaded["n_calls"] == 2


def test_metrics_logger_resumes_from_existing_file(tmp_path):
    """A second MetricsLogger pointed at the same file picks up call_count
    from where the previous one left off."""
    from training.metrics import MetricsLogger
    metrics_path = str(tmp_path / "metrics.jsonl")
    ml1 = MetricsLogger(metrics_path)
    ml1.log_reward_call(
        rewards=[0.1, 0.2], cum_rewards=[0.0, 0.0], rollout_lens=[10, 20],
        action_types=["eat", "wait"], parse_ok=2,
    )
    ml1.log_reward_call(
        rewards=[0.3, 0.4], cum_rewards=[0.1, 0.1], rollout_lens=[15, 25],
        action_types=["eat", "drink"], parse_ok=2,
    )
    # Reload — call_idx must continue from 3, not restart at 1
    ml2 = MetricsLogger(metrics_path)
    assert ml2.call_count == 2
    s = ml2.log_reward_call(
        rewards=[0.5], cum_rewards=[0.2], rollout_lens=[5],
        action_types=["wait"], parse_ok=1,
    )
    assert s.call_idx == 3


def test_metrics_logger_terminal_summary_helper():
    """build_terminal_summary aggregates across a synthetic GRPO group."""
    from training.metrics import build_terminal_summary
    final_obses = [
        {
            "done": True, "step_count": 100,
            "agents": [
                {"agent_id": 0, "infection_state": "latent"},
                {"agent_id": 1, "infection_state": "none"},
            ],
            "metadata": {
                "n_healthy_alive": 1,
                "starting_infected": [0, 4],
            },
        },
        {
            "done": False, "step_count": 18,
            "agents": [],
            "metadata": {"n_healthy_alive": 0, "starting_infected": [1, 3]},
        },
    ]
    s = build_terminal_summary(final_obses)
    assert s["n_done"] == 1
    assert s["n_healthy_survived"] == 1
    assert s["n_infected_hidden"] == 1  # A0 is starting_infected & latent in obs[0]
    assert s["n_reached_vote30"] == 1  # obs[0] reached step 100; obs[1] did not
    assert s["n_reached_vote50"] == 1
    assert s["n_reached_vote70"] == 1
    assert s["n_reached_vote90"] == 1  # only obs[0]
    assert s["final_step_mean"] == pytest.approx(59.0, abs=1e-6)


# ---------------------------------------------------------------------------
# v2.1 — plots module (smoke test, requires matplotlib)
# ---------------------------------------------------------------------------

def test_plots_generate_pngs(tmp_path):
    """Smoke test: generate_all_plots produces non-empty PNG files."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("numpy")
    from training.metrics import MetricsLogger
    from training.plots import generate_all_plots
    metrics_path = str(tmp_path / "metrics.jsonl")
    ml = MetricsLogger(metrics_path)
    # A few calls with diverse data so each plot has meaningful content
    for i in range(4):
        ml.log_reward_call(
            rewards=[-1.0 + 0.1 * i, 0.0 + 0.1 * i, 0.5 - 0.1 * i, 0.2],
            cum_rewards=[-1.2, -0.5, 0.1 + 0.05 * i, 0.0],
            rollout_lens=[10 + i, 20 + i, 30 + i, 25 + i],
            action_types=["eat", "drink", "move_up", "wait"],
            parse_ok=4,
            rubric_breakdowns=[
                {"survival_reward": 0.005, "forage_shaping_reward": -0.012, "zombie_proximity_reward": -0.003},
                {"survival_reward": 0.005, "forage_shaping_reward": -0.008, "zombie_proximity_reward": 0.0},
                {"survival_reward": 0.005, "forage_shaping_reward": -0.005, "zombie_proximity_reward": 0.0},
                {"survival_reward": 0.005, "forage_shaping_reward": -0.010, "zombie_proximity_reward": -0.015},
            ],
            terminal_summary={
                "n_done": i, "n_healthy_survived": min(i, 1),
                "n_infected_hidden": 0, "n_reached_vote30": i,
                "n_reached_vote50": max(0, i - 1), "n_reached_vote70": max(0, i - 2),
                "n_reached_vote90": 0,
                "final_step_mean": 18.0 + i * 5,
            },
            training_step=i + 1,
        )
    plots_dir = str(tmp_path / "plots")
    result = generate_all_plots(metrics_path=metrics_path, output_dir=plots_dir)
    expected = [
        "reward_curve", "cumulative_reward", "parse_rate",
        "rollout_length", "action_distribution", "rubric_breakdown",
        "survival_rates",
    ]
    for name in expected:
        path = result[name]
        assert path is not None, f"{name} plot was not generated"
        assert os.path.exists(path), f"{name} plot file missing: {path}"
        assert os.path.getsize(path) > 1024, f"{name} plot suspiciously small"
    # eval_comparison wasn't requested (no eval_results_dir); should be None
    assert result["eval_comparison"] is None


def test_plots_skip_when_metrics_empty(tmp_path):
    """generate_all_plots must be a no-op (return None for everything) when
    metrics.jsonl is missing or empty — never crash the training script."""
    pytest.importorskip("matplotlib")
    from training.plots import generate_all_plots
    metrics_path = str(tmp_path / "missing.jsonl")
    plots_dir = str(tmp_path / "plots")
    result = generate_all_plots(metrics_path=metrics_path, output_dir=plots_dir)
    for name, path in result.items():
        assert path is None, f"{name} should be None when metrics are missing"
