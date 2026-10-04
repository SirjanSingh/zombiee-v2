"""Prompt text follows the BalanceConfig; bites are visible in the description."""

import hashlib

from survivecity_v2_env.balance import V2_2
from survivecity_v2_env.env import SurviveCityV2Env
from survivecity_v2_env.prompts import build_system_prompt, format_observation_description


def _h(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def test_v2_2_prompts_unchanged():
    # Hashes of the pre-v3 prompt text (recorded before the rules were templated).
    assert _h(build_system_prompt(0, "SIT", prefix_actions=1, balance="v2.2")) == "3b4aae098f406f01"
    assert _h(build_system_prompt(0, "SIT", prefix_actions=5, balance="v2.2")) == "69c615d17eb05a35"


def test_v3_prompt_states_real_rules():
    p = build_system_prompt(0, "SIT", balance="v3-rc1")
    assert "Hunger rises 0.6/step" in p
    assert "move only every 2 steps" in p
    assert "within 4 cells" in p
    assert "tick +1 each step" not in p


def test_wave_line_follows_config():
    p = build_system_prompt(0, "SIT", balance=V2_2.with_(wave_schedule={40: 1}, max_zombies=4))
    assert "Zombie waves spawn at steps 40 (+1, capped at 4 total)." in p


def _desc(**kw):
    agents = [{"agent_id": 0, "row": 7, "col": 7, "hp": 4, "hunger": 0, "thirst": 0,
               "is_alive": True, "locked_out": False}]
    base = dict(agent_id=0, state_dict={"agents": agents, "zombies": []}, phase="p", day_phase="day",
                step=40, broadcasts=[], behavioral_cues=[], last_scan=None, own_inventory=[],
                own_infection_state="none", own_bite_at_step=None, noise_meter=0, noise_threshold=3)
    base.update(kw)
    return format_observation_description(**base)


def test_description_uses_config_numbers():
    d = _desc(balance=V2_2.with_(max_steps=90, hp_max=5))
    assert "Step 40/90" in d and "HP=4/5" in d


def test_bite_events_in_description():
    d = _desc(bite_history=[{"biter_id": 2, "victim_id": 4, "step": 31}])
    assert "A2 bit A4 at t=31" in d
    assert "Bites seen" not in _desc(bite_history=[])


def test_env_description_shows_bites():
    env = SurviveCityV2Env()
    env.reset(seed=0)
    env._episode.bite_history.append({"biter_id": 1, "victim_id": 3, "step": 27})
    obs = env.step({"agent_id": 0, "action_type": "wait"})
    assert "A1 bit A3 at t=27" in obs["description"]
