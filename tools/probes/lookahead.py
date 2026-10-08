"""Rollout (one-step policy improvement) teacher prototype for v3-rc2.

At each A0 turn in the late game (t >= start), try each candidate action, then play the
rest of the episode M times with base policies on a deep-copied env with a reseeded rng,
and pick the action with the best mean score. Earlier turns: base policy.

usage: python lookahead.py N_EPISODES M START [base]
"""
import copy, multiprocessing as mp, random, sys, time, collections
sys.path.insert(0, ".")

ACTS = [("move_up", {}), ("move_down", {}), ("move_left", {}), ("move_right", {}), ("wait", {}),
        ("drink", {}), ("eat", {}), ("pickup", {"item_type": "water"}), ("pickup", {"item_type": "food"})]


def score(env) -> float:
    ep = env._episode
    a0 = ep.agents[0]
    res = ep.extraction_result or {}
    ext = 0 in list(res.get("extracted", []))
    life = a0.death_step if a0.death_step is not None else ep.step_count
    return 1.0 * ext + ALIVE_W * a0.is_alive + 0.002 * life


def finish(env, obs, base, mate, rng):
    while not obs.get("done"):
        aid = obs["metadata"]["current_agent_id"]
        obs = env.step(base(0, obs, rng=rng) if aid == 0 else mate(aid, obs, rng=rng))
    return score(env)


def candidates(obs):
    me = next(a for a in obs["agents"] if a["agent_id"] == 0)
    inv = me.get("inventory") or []
    out = []
    for t, k in ACTS:
        if t == "drink" and "water" not in inv and (me["row"], me["col"]) not in WATER:
            continue
        if t == "eat" and "food" not in inv and (me["row"], me["col"]) not in FOOD:
            continue
        if t == "pickup" and (len(inv) >= 3 or (me["row"], me["col"]) not in (WATER if k["item_type"] == "water" else FOOD)):
            continue
        if t.startswith("move"):
            dr, dc = MOVES[t]
            if not _walk(me["row"] + dr, me["col"] + dc):
                continue
        out.append((t, k))
    return out


def resample_hidden(ep, rng):
    """Belief sample for a rollout: what A0 cannot observe is redrawn.

    - episode_seed drives the hash-deterministic bite and cue outcomes -> redraw.
    - infection of A1-A4 is private; keep agents publicly seen biting (bite_history)
      as they are, permute the infection fields among the other living teammates.
    """
    ep.episode_seed = rng.randrange(10 ** 9)
    known = {b["biter_id"] for b in ep.bite_history}
    pool = [a for a in ep.agents[1:] if a.is_alive and a.agent_id not in known]
    fields = [(a.infection_state, a.infection_role, a.bite_at_step) for a in pool]
    rng.shuffle(fields)
    for a, (st, role, bs) in zip(pool, fields):
        a.infection_state, a.infection_role, a.bite_at_step = st, role, bs


FAIR = False
import os
ALIVE_W = float(os.environ.get("ALIVE_W", "0.1"))


def choose(env, obs, base, mate, M, seed_key):
    best, best_v = None, -1e9
    for i, (t, k) in enumerate(candidates(obs)):
        v = 0.0
        for j in range(M):
            e = copy.deepcopy(env)
            e._episode.rng = random.Random(f"{seed_key}|{j}")      # common random numbers across actions
            if FAIR:
                resample_hidden(e._episode, random.Random(f"{seed_key}|{j}|h"))
            o = e.step({"agent_id": 0, "action_type": t, **k})
            v += finish(e, o, base, mate, random.Random(f"{seed_key}|{j}|p"))
        v /= M
        if v > best_v:
            best, best_v = (t, k), v
    return {"agent_id": 0, "action_type": best[0], **best[1]}


def run(args):
    global FAIR
    seed, M, start, base_name, FAIR = args
    from survivecity_v2_env.balance import get_balance
    from survivecity_v2_env.env import SurviveCityV2Env
    from training.policies import get_policy
    base, mate = get_policy(base_name), get_policy("camp")
    env = SurviveCityV2Env(balance=get_balance("v3-rc2"), a0_healthy=True)
    obs = env.reset(seed=seed)
    rng = random.Random(f"eval|{seed}")
    while not obs.get("done"):
        aid = obs["metadata"]["current_agent_id"]
        if aid == 0:
            act = (choose(env, obs, base, mate, M, f"{seed}|{obs['step_count']}") if obs["step_count"] >= start
                   else base(0, obs, rng=rng))
        else:
            act = mate(aid, obs, rng=rng)
        obs = env.step(act)
    ep = env._episode
    a0 = ep.agents[0]
    res = ep.extraction_result or {}
    return {"a0_ext": 0 in list(res.get("extracted", [])), "team": bool(res.get("success")),
            "life": a0.death_step if a0.death_step is not None else ep.step_count,
            "cause": a0.death_cause if not a0.is_alive else "alive"}


from training.policies import MOVES, _walk, WATER_CELLS as WATER, FOOD_CELLS as FOOD  # noqa: E402

if __name__ == "__main__":
    N, M, START = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    base = sys.argv[4] if len(sys.argv) > 4 else "camp_v2"
    fair = len(sys.argv) > 5 and sys.argv[5] == "fair"
    rng = random.Random(999)
    seeds = [rng.randint(0, 999999) for _ in range(N)]
    t = time.time()
    with mp.Pool(11) as pool:
        recs = pool.map(run, [(s, M, START, base, fair) for s in seeds])
    n = len(recs)
    print(f"lookahead M={M} start={START} base={base} fair={fair} N={n}: A0 extracted {sum(r['a0_ext'] for r in recs)/n:.1%} "
          f"team {sum(r['team'] for r in recs)/n:.1%} life {sum(r['life'] for r in recs)/n:.1f} "
          f"{dict(collections.Counter(r['cause'] for r in recs).most_common())} ({time.time()-t:.0f}s)")
