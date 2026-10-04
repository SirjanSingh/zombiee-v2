"""BalanceConfig: v2.2 preset reproduces the original game, and each knob takes effect."""

import hashlib
import json
import random

import pytest

from survivecity_v2_env.balance import (
    V2_2, BalanceConfig, get_balance, meter_tick,
)
from survivecity_v2_env.env import SurviveCityV2Env
from survivecity_v2_env.game import (
    advance_step, apply_agent_action, create_episode, _find_nearest_agent_for_zombie,
)

_ACTIONS = ["move_up", "move_down", "move_left", "move_right", "wait", "eat", "drink",
            "pickup", "scan"]


def _trajectory_hash(balance, n_eps=8) -> str:
    """Hash every step of a few seeded random-policy episodes."""
    h = hashlib.sha256()
    for sd in range(n_eps):
        env = SurviveCityV2Env(balance=balance)
        obs = env.reset(seed=sd * 7919)
        rng = random.Random(sd)
        while not obs.get("done"):
            aid = obs["metadata"]["current_agent_id"]
            obs = env.step({"agent_id": aid, "action_type": rng.choice(_ACTIONS),
                            "scan_target": rng.randrange(5)})
            ep = env._episode
            h.update(json.dumps([
                ep.step_count, round(obs.get("reward") or 0.0, 9),  # 9 dp: float sum() differs in last bits across Python 3.11/3.12+
                [(a.row, a.col, a.hp, a.hunger, a.thirst, a.is_alive, a.infection_state)
                 for a in ep.agents],
                [(z.row, z.col) for z in ep.zombies],
            ]).encode())
    return h.hexdigest()[:16]


def test_v2_2_preset_trajectories_unchanged():
    # Golden hash recorded when BalanceConfig was introduced (behaviour-identical
    # refactor, verified against pre-refactor heuristic/random/oracle traces).
    assert _trajectory_hash("v2.2") == V2_2_GOLDEN


V2_2_GOLDEN = "da750a6eb61e0520"


def test_meter_tick_integer_rates():
    assert [meter_tick(t, 1.0) for t in range(6)] == [1] * 6
    assert [meter_tick(t, 2.0) for t in range(4)] == [2] * 4


def test_meter_tick_matches_v2_infected_rule():
    # v2.2 hard-coded infected hunger as +2 on even steps, +1 on odd steps.
    assert [meter_tick(t, 1.5) for t in range(8)] == [2 if t % 2 == 0 else 1 for t in range(8)]


def test_meter_tick_fractional_rate_averages_out():
    assert sum(meter_tick(t, 0.75) for t in range(100)) == 75


def test_get_balance_resolves_presets():
    assert get_balance("v2.2") is V2_2
    cfg = BalanceConfig(p_bite=0.1)
    assert get_balance(cfg) is cfg
    with pytest.raises(ValueError):
        get_balance("nope")


def test_with_accepts_wave_dict():
    cfg = V2_2.with_(wave_schedule={50: 1, 25: 4})
    assert cfg.wave_schedule == ((25, 4), (50, 1))
    assert cfg.waves == {25: 4, 50: 1}


def test_episode_carries_balance():
    cfg = V2_2.with_(max_steps=40, hp_max=5)
    ep = create_episode(seed=1, balance=cfg)
    assert ep.balance is cfg
    assert ep.max_steps == 40
    assert all(a.hp == 5 for a in ep.agents)


def test_starve_threshold_knob():
    ep = create_episode(seed=0, balance=V2_2.with_(starve_threshold=3))
    a = ep.agents[0]
    a.infection_state, a.infection_role = "none", None
    a.row, a.col = 0, 7  # outside safehouse: no healing
    for _ in range(3):
        apply_agent_action(ep, 0, "wait")
    assert a.hunger == 3 and a.hp == 2


def test_thirst_rate_knob():
    ep = create_episode(seed=0, balance=V2_2.with_(thirst_rate=0.5))
    a = ep.agents[0]
    for t in range(10):
        ep.step_count = t
        apply_agent_action(ep, 0, "wait")
    assert a.thirst == 5


def test_zombie_chase_radius_knob():
    ep = create_episode(seed=0, balance=V2_2.with_(zombie_chase_radius=3))
    z = ep.zombies[0]
    z.row, z.col = 0, 0
    for a in ep.agents[1:]:
        a.is_alive = False
    a0 = ep.agents[0]
    a0.row, a0.col = 3, 3   # distance 6 > radius
    assert _find_nearest_agent_for_zombie(z, ep) is None
    a0.row, a0.col = 1, 2   # distance 3 <= radius
    assert _find_nearest_agent_for_zombie(z, ep) == (1, 2)


def test_wave_schedule_and_cap_knobs():
    ep = create_episode(seed=0, balance=V2_2.with_(wave_schedule={5: 4}, max_zombies=5))
    n0 = len(ep.zombies)
    for _ in range(5):
        advance_step(ep)
    assert len(ep.zombies) == 5 == n0 + 2   # wanted 4, capped at 5 total
    for _ in range(30):
        advance_step(ep)
    assert len(ep.zombies) == 5             # no v2.2 waves at 25


def test_p_bite_zero_never_bites():
    env = SurviveCityV2Env(balance=V2_2.with_(p_bite=0.0))
    obs = env.reset(seed=3)
    while not obs.get("done"):
        aid = obs["metadata"]["current_agent_id"]
        obs = env.step({"agent_id": aid, "action_type": "wait"})
    assert env._episode.bite_history == []


def test_reveal_steps_knob():
    ep = create_episode(seed=0, balance=V2_2.with_(biter_reveal_step=3))
    biter = next(a for a in ep.agents if a.infection_role == "biter")
    for _ in range(3):
        advance_step(ep)
    assert biter.infection_state == "revealed"


def test_zombie_move_every_knob():
    ep = create_episode(seed=0, balance=V2_2.with_(zombie_move_every=2))
    from survivecity_v2_env.game import advance_zombies
    start = [(z.row, z.col) for z in ep.zombies]
    ep.step_count = 1                      # odd step: zombies hold still
    advance_zombies(ep)
    assert [(z.row, z.col) for z in ep.zombies] == start
    ep.step_count = 2
    advance_zombies(ep)
    assert [(z.row, z.col) for z in ep.zombies] != start


def test_starting_infected_progression_flag():
    for flag, expect_dead in ((True, True), (False, False)):
        ep = create_episode(seed=0, balance=V2_2.with_(starting_infected_progression=flag))
        biter = next(a for a in ep.agents if a.infection_role == "biter")
        biter.hunger = biter.thirst = -10**6   # keep it fed
        for _ in range(31):
            advance_step(ep)
        assert (biter.death_cause == "infection_progression") == expect_dead


def test_hp_max_above_3_builds_observation():
    env = SurviveCityV2Env(balance=V2_2.with_(hp_max=5))
    obs = env.reset(seed=0)
    assert obs["agents"][0]["hp"] == 5
