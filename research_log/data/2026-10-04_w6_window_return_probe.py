import sys, logging, random, statistics as S; logging.disable(50)
sys.path.insert(0, r"D:\projs\extra\v2\zombiee-v2")
from training.scenarios import build_scenarios, replay_to
from training.policies import camp_action, heuristic_v3_action, random_policy
K, H = 5, 25
A0 = {"camp": lambda o, r: camp_action(0, o), "heuristic_v3": lambda o, r: heuristic_v3_action(0, o, rng=r),
      "wait": lambda o, r: {"agent_id": 0, "action_type": "wait"}, "random": lambda o, r: random_policy(0, o, rng=r)}
def run(sc, m):
    env, obs, ok = replay_to(sc["seed"], sc["t"], sc["mix"]); r = random.Random(1)
    c0 = obs["metadata"]["cumulative_rewards"][0]; used = 0; t_end = sc["t"] + K + H
    while not obs.get("done") and env._episode.step_count < t_end:
        aid = obs["metadata"]["current_agent_id"]
        if aid == 0 and used < K: obs = env.step(A0[m](obs, r)); used += 1
        else: obs = env.step(camp_action(aid, obs))
    a = env._episode.agents[0]
    return obs["metadata"]["cumulative_rewards"][0] - c0, a.is_alive, (a.hp if a.is_alive else 0)
scen = build_scenarios(64, seed=11, prefix_k=K)
res = {m: [run(sc, m) for sc in scen] for m in A0}
for m, rows in res.items():
    print(f"{m:<13} shaped {S.mean(x[0] for x in rows):+.3f}  alive_end {sum(x[1] for x in rows)}/64  hp_end {S.mean(x[2] for x in rows):.2f}")
def wins(a, b, i): return sum(x[i] > y[i] for x, y in zip(res[a], res[b])), sum(x[i] < y[i] for x, y in zip(res[a], res[b]))
for a, b in (("camp", "wait"), ("camp", "random"), ("wait", "random")):
    print(f"{a} vs {b}: shaped win/loss {wins(a,b,0)}  alive win/loss {wins(a,b,1)}  hp win/loss {wins(a,b,2)}")
