"""Soft-label dataset for Laya from the fair lookahead teacher (v3-rc2), CPU only.

Episodes are driven by the lookahead teacher itself (camp_v2 before START), with probability EPS
of a random legal action instead (DART-style noise, so off-path states get labels too). Every A0
state is written as a Laya row:
  t <  START: hard label = camp_v2's action
  t >= START: camp_v2's action, overridden by the lookahead's best action only when that beats
              camp_v2's own action value by more than DELTA (M fair rollouts per action). Early in an
              episode all values tie (~0.1), so soft/argmax labels there are pure rollout noise.

usage: python tools/probes/lookahead_labels.py N_EPISODES OUT.jsonl [--m 16 --start 55 --eps 0.2 --tau 0.05 --workers 60]
"""
import argparse, json, math, multiprocessing as mp, os, random, sys, time, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def episode(args):
    seed, m, start, eps, tau, delta = args
    import lookahead as LA
    LA.FAIR, LA.ALIVE_W = True, 0.0
    from survivecity_v2_env.balance import get_balance
    from survivecity_v2_env.env import SurviveCityV2Env
    from training.policies import get_policy
    from training.tiny_policy import ACTIONS, action_label, legal_mask, to_action
    from training.laya_policy import QUESTION, QID, state_text
    base, mate = get_policy("camp_v2"), get_policy("camp")
    env = SurviveCityV2Env(balance=get_balance("v3-rc2"), a0_healthy=True)
    obs = env.reset(seed=seed)
    rng = random.Random(f"lal|{seed}")
    rows = []
    while not obs.get("done"):
        aid = obs["metadata"]["current_agent_id"]
        if aid != 0:
            obs = env.step(mate(aid, obs, rng=rng))
            continue
        t = obs["step_count"]
        if t < start:
            teach = base(0, obs, rng=rng)
            lab = action_label(teach)
            if lab is not None:
                rows.append({"state": state_text(obs), "questions": QUESTION, "expected": {QID: ACTIONS[lab]}, "t": t})
        else:
            vals = LA.action_values(env, obs, base, mate, m, f"lal|{seed}|{t}")
            vmap = {ACTIONS[action_label({"action_type": a, **k})]: v for a, k, v in vals}
            b = base(0, obs, rng=rng)
            b_lab = action_label(b)
            best = max(vals, key=lambda x: x[2])
            best_lab = ACTIONS[action_label({"action_type": best[0], **best[1]})]
            b_name = ACTIONS[b_lab] if b_lab is not None else None
            override = b_name is None or best[2] - vmap.get(b_name, -1.0) > delta
            label = best_lab if override else b_name
            rows.append({"state": state_text(obs), "questions": QUESTION, "expected": {QID: label}, "t": t,
                         "override": override, "base": b_name, "values": {k: round(v, 4) for k, v in vmap.items()}})
            teach = to_action(ACTIONS.index(label), obs)
        if rng.random() < eps:
            mask = legal_mask(obs)
            act = to_action(rng.choice([i for i in range(len(ACTIONS)) if mask[i]]), obs)
        else:
            act = teach
        obs = env.step(act)
    ep = env._episode
    res = ep.extraction_result or {}
    return rows, {"a0_ext": 0 in list(res.get("extracted", [])), "alive": ep.agents[0].is_alive}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("n", type=int); p.add_argument("out")
    p.add_argument("--m", type=int, default=16); p.add_argument("--start", type=int, default=55)
    p.add_argument("--eps", type=float, default=0.2); p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=60); p.add_argument("--seed", type=int, default=7070)
    a = p.parse_args()
    r = random.Random(a.seed)
    jobs = [(r.randint(0, 999999), a.m, a.start, a.eps, a.tau, a.delta) for _ in range(a.n)]
    t0 = time.time(); n_rows = n_soft = 0; recs = []
    with mp.get_context("spawn").Pool(a.workers) as pool, open(a.out, "w", encoding="utf-8") as f:
        for i, (rows, rec) in enumerate(pool.imap_unordered(episode, jobs), 1):
            recs.append(rec)
            for row in rows:
                f.write(json.dumps(row) + "\n")
                n_rows += 1; n_soft += bool(row.get("override"))
            if i % 25 == 0:
                print(f"{i}/{a.n} episodes, {n_rows} rows ({n_soft} overrides), {time.time()-t0:.0f}s", flush=True)
    print(f"done: {n_rows} rows ({n_soft} overrides) from {a.n} episodes; driver A0 extracted "
          f"{sum(x['a0_ext'] for x in recs)/len(recs):.1%} alive {sum(x['alive'] for x in recs)/len(recs):.1%} ({time.time()-t0:.0f}s)")
