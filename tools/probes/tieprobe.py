import sys, random, statistics as st, logging; logging.disable(logging.CRITICAL); sys.path.insert(0,'.')
from training.scenarios import build_scenarios, replay_to, score_completion, ReplayCache, parse_scenario_tag
from training.policies import get_policy
sc = build_scenarios(24, seed=11, balance="v3-rc1")
def plan(tag, seed, k=5):
    s,t,m = parse_scenario_tag(tag); env,obs,ok = replay_to(s,t,m,balance="v3-rc1")
    pol=get_policy("random"); roll=get_policy("camp"); acts=[]; rng=random.Random(seed)
    while len(acts)<k and not obs.get("done"):
        aid=obs["metadata"]["current_agent_id"]
        a = pol(aid,obs,rng=rng) if aid==0 else roll(aid,obs,rng=random.Random(1))
        if aid==0: acts.append(a)
        obs=env.step(a)
    return acts
for mode in ("survival","graded"):
    cache=ReplayCache(); ties=0; stds=[]
    for s in sc:
        r=[score_completion(s["tag"],plan(s["tag"],g),horizon=25,balance="v3-rc1",window_return=mode,step1_weight=0.0,cache=cache)["reward"] for g in range(8)]
        sd=st.pstdev(r); stds.append(sd); ties+= sd<1e-6
    print(f"{mode}: groups with zero spread {ties}/24, mean std {st.mean(stds):.3f}")
