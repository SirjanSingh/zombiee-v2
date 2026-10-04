"""W6 mid-episode scenarios: deterministic replay, healthy A0, decision density, scoring."""

import random

from survivecity_v2_env.balance import get_balance
from survivecity_v2_env.env import SurviveCityV2Env
from survivecity_v2_env.game import create_episode
from training.policies import MixPolicy, sample_mix
from training import scenarios as S


def _sig(env):
    ep = env._episode
    return (
        ep.step_count,
        [(a.row, a.col, a.hp, a.hunger, a.thirst, a.is_alive, a.infection_state, a.infection_role,
          tuple(a.inventory), a.locked_out, a.bite_at_step) for a in ep.agents],
        [(z.zombie_id, z.row, z.col) for z in ep.zombies],
        sorted(ep.food_present.items()), sorted(ep.food_respawn_at.items()),
        sorted(ep.medicine_present.items()), sorted(ep.medicine_respawn_at.items()),
        list(map(lambda b: tuple(sorted(b.items())), ep.bite_history)),
        sorted(ep.lockout_results.items()), ep.noise_meter, ep.rng.getstate(),
    )


def test_replay_reproduces_exact_state_50_random():
    rng = random.Random(123)
    checked = 0
    while checked < 50:
        seed, mix = rng.randint(0, 999999), sample_mix(rng)
        t = rng.randint(0, 80)
        env = SurviveCityV2Env(a0_healthy=True)
        obs = env.reset(seed=seed)
        pol = MixPolicy(mix, seed)
        sig = None
        while not obs.get("done"):
            if env._episode.step_count == t and obs["metadata"]["current_agent_id"] == 0:
                sig = _sig(env)
                desc = obs["description"]
                break
            if env._episode.step_count > t:
                break
            obs = env.step(pol(obs["metadata"]["current_agent_id"], obs))
        if sig is None:
            continue   # A0 dead or episode over before t: not a valid scenario
        env2, obs2, ok = S.replay_to(seed, t, mix)
        assert ok
        assert _sig(env2) == sig, (seed, t, mix)
        assert obs2["description"] == desc
        checked += 1


def test_a0_healthy_option():
    for seed in range(200):
        ep = create_episode(seed=seed, a0_healthy=True)
        assert ep.agents[0].infection_role is None
        assert sum(a.infection_role is not None for a in ep.agents) == 2
    # Default draw unchanged (golden hash test covers trajectories).
    assert any(create_episode(seed=s).agents[0].infection_role for s in range(50))


def test_tag_roundtrip():
    tag = S.scenario_tag(12, 34, "chech")
    assert S.parse_scenario_tag("xx " + tag + "\nStep") == (12, 34, "chech")
    assert S.parse_scenario_tag("[SEED:7]\nStep 0") == (7, 0, None)


def test_build_scenarios_caps_routine_and_replays():
    sc = S.build_scenarios(40, seed=1)
    assert len(sc) == 40
    st = S.dataset_stats(sc)
    assert st["routine_frac"] <= 0.35
    assert len(st["t_hist_by_10"]) >= 3          # not all at t=0
    for s in sc[:10]:
        env, obs, ok = S.replay_to(s["seed"], s["t"], s["mix"])
        assert ok and obs["description"] == s["description"]


def test_score_completion_deterministic_and_window_bounded():
    sc = S.build_scenarios(4, seed=2)
    acts = [{"agent_id": 0, "action_type": "wait"}] * 5
    cache = S.ReplayCache()
    for s in sc:
        prompt = s["tag"] + "\n" + s["description"]
        r1 = S.score_completion(prompt, acts, horizon=10, cache=cache)
        r2 = S.score_completion(prompt, acts, horizon=10)
        assert r1["reward"] == r2["reward"]
        assert r1["t_end"] <= s["t"] + 5 + 10
        assert len(r1["gigpo_entries"]) == len(r1["model_raws"]) <= 5


def test_invalid_penalty_applies():
    s = S.build_scenarios(1, seed=3)[0]
    prompt = s["tag"]
    acts = [{"agent_id": 0, "action_type": "wait"}]
    ok = S.score_completion(prompt, acts, parse_ok=True)["reward"]
    bad = S.score_completion(prompt, acts, parse_ok=False, invalid_action_penalty=0.1)["reward"]
    assert abs((ok - bad) - 0.1) < 1e-9


def test_survival_window_return_values():
    sc = S.build_scenarios(6, seed=4)
    acts = [{"agent_id": 0, "action_type": "wait"}] * 5
    for s in sc:
        r = S.score_completion(s["tag"], acts, horizon=10, window_return="survival", step1_weight=0.0)
        w = r["window_return"]
        assert w == -1.0 or 0.5 <= w <= 1.5
        assert r["reward"] == w   # step1_weight 0, parse ok, no format bonus
        assert all(e[1] == w for e in r["gigpo_entries"])
