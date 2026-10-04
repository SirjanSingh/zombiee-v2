"""W4 radio + extraction objective (preset v3-rc2).

Off in v2.2 / v3-rc1 (pinned hashes below), on in v3-rc2: radio zone from a
separate rng, helicopter at extraction_step, contamination rule, rewards routed
through pending_reward into cumulative_rewards, radio text only in v3-rc2.
"""

import hashlib
import os
import random
import sys

from survivecity_v2_env.balance import V3_RC1, V3_RC2, get_balance
from survivecity_v2_env.env import SurviveCityV2Env
from survivecity_v2_env.extraction import CORNERS, pick_zone, zone_cells
from survivecity_v2_env.game import advance_step, create_episode
from survivecity_v2_env.prompts import build_system_prompt

sys.path.insert(0, os.path.dirname(__file__))
from test_balance import _trajectory_hash  # noqa: E402


def _h(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# --- extraction off: nothing changed -------------------------------------------------

def test_v3_rc1_trajectories_unchanged():
    # Recorded on ac57d03, before W4 touched the game.
    assert _trajectory_hash("v3-rc1") == "2cb60fac677863fd"


def test_v3_rc1_prompts_unchanged():
    assert _h(build_system_prompt(0, "SIT", prefix_actions=1, balance="v3-rc1")) == "0302c2862d148c87"
    assert _h(build_system_prompt(0, "SIT", prefix_actions=5, balance="v3-rc1")) == "b4298d440adb4ab4"


def test_presets():
    assert not get_balance("v2.2").extraction_enabled and not V3_RC1.extraction_enabled
    assert V3_RC2.extraction_enabled and V3_RC2.max_steps == 90
    assert (V3_RC2.radio_step, V3_RC2.extraction_step) == (60, 90)
    # v3-rc2 = v3-rc1 + extraction only
    same = {k: v for k, v in V3_RC1.to_dict().items()
            if k not in ("max_steps", "extraction_enabled", "radio_step", "extraction_step")}
    assert all(V3_RC2.to_dict()[k] == v for k, v in same.items())


def test_no_extraction_metadata_when_off():
    obs = SurviveCityV2Env(balance="v3-rc1").reset(seed=1)
    assert "extraction" not in obs["metadata"] and "extraction_zone" not in obs["metadata"]


# --- zone geometry and the radio rng -------------------------------------------------

def test_zone_cells():
    assert zone_cells("NE", 2) == [(0, 12), (0, 13), (0, 14), (1, 12), (1, 13), (1, 14), (2, 13), (2, 14)]
    for name in CORNERS:
        cells = zone_cells(name, 2)
        assert len(cells) == 8                       # 3x3 minus the corner choke wall


def test_radio_rng_independent_of_episode_rng():
    """Same seed -> identical episode up to the radio with extraction on or off."""
    on, off = V3_RC2, V3_RC2.with_(extraction_enabled=False)
    for seed in (3, 17, 4242):
        envs = [SurviveCityV2Env(balance=b) for b in (on, off)]
        obs = [e.reset(seed=seed) for e in envs]
        rng = random.Random(seed)
        while obs[0]["step_count"] < on.radio_step and not obs[0]["done"]:
            aid = obs[0]["metadata"]["current_agent_id"]
            act = {"agent_id": aid, "action_type": rng.choice(["wait", "move_up", "move_left", "drink", "eat"])}
            obs = [e.step(dict(act)) for e in envs]
            a, b = (e._episode for e in envs)
            assert [(x.row, x.col, x.hp, x.hunger, x.thirst, x.is_alive, x.infection_state) for x in a.agents] == \
                   [(x.row, x.col, x.hp, x.hunger, x.thirst, x.is_alive, x.infection_state) for x in b.agents]
            assert [(z.row, z.col) for z in a.zombies] == [(z.row, z.col) for z in b.zombies]
        assert envs[0]._episode.rng.getstate() == envs[1]._episode.rng.getstate()


def test_zone_pick_is_seeded_and_covers_corners():
    assert pick_zone(5) == pick_zone(5)
    assert {pick_zone(s) for s in range(200)} == set(CORNERS)


def test_radio_announces_zone_in_metadata():
    ep = create_episode(seed=9, balance=V3_RC2)
    for _ in range(V3_RC2.radio_step - 1):
        advance_step(ep)
    assert ep.extraction_zone_name is None
    advance_step(ep)
    assert ep.extraction_zone_name == pick_zone(9)
    assert ep.extraction_zone == zone_cells(pick_zone(9), 2)


# --- helicopter + rewards -------------------------------------------------------------

def _ep_at_extraction(seed=0):
    """v3-rc2 episode one step before the helicopter, everyone fed, A1/A2 infected."""
    ep = create_episode(seed=seed, balance=V3_RC2)
    while ep.step_count < V3_RC2.extraction_step - 1:
        for a in ep.agents:
            a.hunger = a.thirst = 0
        advance_step(ep)
    for a in ep.agents:
        a.is_alive, a.hp = True, 3
        a.infection_state, a.infection_role, a.pending_reward = "none", None, 0.0
    for i in (1, 2):
        ep.agents[i].infection_state = "latent"
    for a in ep.agents:
        a.row, a.col = 7, 7                          # everyone in the safehouse
    return ep


def _put_in_zone(ep, *ids):
    cells = ep.extraction_zone
    for k, i in enumerate(ids):
        ep.agents[i].row, ep.agents[i].col = cells[k]


def test_successful_extraction_rewards():
    ep = _ep_at_extraction()
    _put_in_zone(ep, 0, 3)
    advance_step(ep)
    res = ep.extraction_result
    assert ep.done and res["flew"] and res["success"] and not res["failed_flight"]
    assert res["extracted"] == [0, 3] and res["n_extracted"] == 2
    cfg = V3_RC2
    assert ep.agents[0].pending_reward == cfg.extract_reward + cfg.extract_team_bonus
    assert ep.agents[3].pending_reward == cfg.extract_reward + cfg.extract_team_bonus
    assert ep.agents[4].pending_reward == 0.0         # healthy, not aboard
    assert ep.agents[1].pending_reward == 0.0         # infected lose
    assert ep.agents[2].pending_reward == 0.0


def test_contamination_fails_flight():
    ep = _ep_at_extraction()
    _put_in_zone(ep, 0, 3, 1)                         # A1 is latent-infected
    advance_step(ep)
    res = ep.extraction_result
    assert res["failed_flight"] and not res["success"] and res["extracted"] == []
    assert res["infected_aboard"] == [1]
    cfg = V3_RC2
    assert ep.agents[0].pending_reward == ep.agents[3].pending_reward == cfg.failed_flight_penalty
    assert ep.agents[4].pending_reward == 0.0
    assert ep.agents[1].pending_reward == ep.agents[2].pending_reward == cfg.infected_win_reward


def test_nobody_extracted_pays_infected():
    ep = _ep_at_extraction()
    advance_step(ep)
    assert ep.extraction_result["n_extracted"] == 0 and not ep.extraction_result["failed_flight"]
    assert ep.agents[1].pending_reward == V3_RC2.infected_win_reward
    assert ep.agents[0].pending_reward == 0.0


def test_dead_agents_do_not_board():
    ep = _ep_at_extraction()
    _put_in_zone(ep, 0, 1)
    ep.agents[1].is_alive = False                     # dead infected body in the zone
    advance_step(ep)
    assert ep.extraction_result["success"] and ep.extraction_result["aboard"] == [0]


def _run(balance, seed=0, steps=None, setup=None):
    """Drive the env with waits (keeping everyone fed); return final cumulative_rewards."""
    env = SurviveCityV2Env(balance=balance)
    obs = env.reset(seed=seed)
    ep = env._episode
    for a in ep.agents:
        a.infection_state, a.infection_role, a.bite_at_step = "none", None, None
    ep.agents[4].infection_state, ep.agents[4].infection_role = "latent", "saboteur"
    while not obs["done"]:
        for a in ep.agents:
            a.hunger = a.thirst = 0
        if setup is not None:
            setup(ep)
        aid = obs["metadata"]["current_agent_id"]
        obs = env.step({"agent_id": aid, "action_type": "wait"})
    return obs["metadata"]["cumulative_rewards"], obs


def test_rewards_land_in_cumulative_rewards():
    """Milestones + extraction reach the right agents through the env reward path."""
    base = V3_RC2.with_(max_steps=8, radio_step=2, extraction_step=8, milestone_steps=(3, 5))
    zero = base.with_(milestone_reward=0.0, extract_reward=0.0, extract_team_bonus=0.0,
                      failed_flight_penalty=0.0, infected_win_reward=0.0)

    def board_a0(ep):
        ep.zombies.clear()                              # corner zombies would kill A0 in the zone
        if ep.extraction_zone:
            ep.agents[0].row, ep.agents[0].col = ep.extraction_zone[0]

    on, obs = _run(base, setup=board_a0)
    off, _ = _run(zero, setup=board_a0)
    d = {i: round(on[i] - off[i], 6) for i in on}
    assert obs["metadata"]["extraction"]["result"]["extracted"] == [0]
    assert d[0] == round(2 * base.milestone_reward + base.extract_reward, 6)
    for i in (1, 2, 3):
        assert d[i] == round(2 * base.milestone_reward, 6)
    assert d[4] == 0.0                                  # infected: no milestones, flight succeeded


# --- prompts --------------------------------------------------------------------------

def test_prompt_radio_text_only_in_rc2():
    assert "EXTRACTION" in build_system_prompt(0, "SIT", balance="v3-rc2")
    assert "EXTRACTION" not in build_system_prompt(0, "SIT", balance="v3-rc1")

    env = SurviveCityV2Env(balance="v3-rc1")
    assert "radio" not in env.reset(seed=2)["description"].lower()

    env = SurviveCityV2Env(balance="v3-rc2")
    obs = env.reset(seed=2)
    assert "Survive; a radio message will come at t=60." in obs["description"]
    assert obs["metadata"]["extraction_zone"] == []
    ep = env._episode
    for a in ep.agents:
        a.infection_state, a.infection_role, a.bite_at_step = "none", None, None
    for _ in range(2000):
        if ep.step_count >= 61 or obs["done"]:
            break
        ep.zombies.clear()
        for a in ep.agents:
            a.hunger = a.thirst = 0
        obs = env.step({"agent_id": obs["metadata"]["current_agent_id"], "action_type": "wait"})
    assert ep.step_count == 61 and not obs["done"]
    name = pick_zone(2)
    assert f"RADIO: extraction at the {name} corner" in obs["description"]
    assert "helicopter loads at t=90, 29 turns left. Anyone infected aboard dooms the flight." in obs["description"]
    assert obs["metadata"]["extraction"]["zone_name"] == name
    assert obs["metadata"]["extraction_zone"] == [list(c) for c in zone_cells(name, 2)]
