"""How noisy are the lookahead teacher's labels? Replays recorded Laya episodes and, at every A0 state from
t>=START, computes the fair action values twice with independent rollout seeds (M each). Reports argmax
agreement, override (delta-rule) agreement and the value gap distribution.

usage: python tools/probes/label_noise.py REC.jsonl --episodes 40 --m 16 --workers 40
"""
import argparse, json, multiprocessing as mp, os, random, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def work(job):
    seed, actions, m, start, delta = job
    import lookahead as LA
    LA.FAIR, LA.ALIVE_W = True, 0.0
    from survivecity_v2_env.balance import get_balance
    from survivecity_v2_env.env import SurviveCityV2Env
    from training.policies import get_policy
    base, mate = get_policy("camp_v2"), get_policy("camp")
    env = SurviveCityV2Env(balance=get_balance("v3-rc2"), a0_healthy=True)
    obs = env.reset(seed=seed)
    rng = random.Random(f"laya|{seed}")
    lrng = random.Random(f"noise|{seed}")
    out, turn = [], 0
    while not obs.get("done"):
        aid = obs["metadata"]["current_agent_id"]
        if aid != 0:
            obs = env.step(mate(aid, obs, rng=rng)); continue
        t = obs["step_count"]
        if t >= start:
            b = base(0, obs, rng=lrng)["action_type"]
            res = []
            for k in ("A", "B"):
                v = LA.action_values(env, obs, base, mate, m, f"noise|{seed}|{t}|{k}")
                best = max(v, key=lambda x: x[2])
                vb = max((x[2] for x in v if x[0] == b), default=-1)
                srt = sorted(x[2] for x in v)
                res.append((best[0], best[2] - vb > delta, srt[-1] - (srt[-2] if len(srt) > 1 else 0)))
            out.append({"t": t, "argmax_agree": res[0][0] == res[1][0], "override_A": res[0][1], "override_B": res[1][1],
                        "gap": res[0][2]})
        act = dict(actions[turn]) if turn < len(actions) else {"agent_id": 0, "action_type": "wait"}
        turn += 1
        obs = env.step(act)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("rec"); p.add_argument("--episodes", type=int, default=40); p.add_argument("--m", type=int, default=16)
    p.add_argument("--start", type=int, default=55); p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=40)
    a = p.parse_args()
    recs = [json.loads(l) for l in open(a.rec)][:a.episodes]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        rows = [r for rs in pool.map(work, [(x["seed"], x["a0_actions"], a.m, a.start, a.delta) for x in recs]) for r in rs]
    n = len(rows)
    agree = sum(r["argmax_agree"] for r in rows) / n
    ov = [r for r in rows if r["override_A"] or r["override_B"]]
    ov_agree = sum(r["override_A"] == r["override_B"] for r in rows) / n
    gaps = sorted(r["gap"] for r in rows)
    print(json.dumps({"states": n, "m": a.m, "argmax_agreement": round(agree, 3), "override_rule_agreement": round(ov_agree, 3),
                      "states_overridden_by_either": len(ov), "both_override": sum(r["override_A"] and r["override_B"] for r in rows),
                      "top2_gap_p25_p50_p75": [round(gaps[n // 4], 3), round(gaps[n // 2], 3), round(gaps[3 * n // 4], 3)]}))
