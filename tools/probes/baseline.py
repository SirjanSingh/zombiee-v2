import random, logging, collections, time, sys, statistics as S
logging.disable(logging.CRITICAL)
import os; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..'))
from survivecity_v2_env.env import SurviveCityV2Env
from training.inference import forage_heuristic_action

def run(policy, n=60, seed=42, a0_policy=None):
    rng=random.Random(seed); res=[]; causes=collections.Counter(); a0c=collections.Counter()
    for ep in range(n):
        env=SurviveCityV2Env(); s=rng.randint(0,999999)
        obs=env.reset(seed=s); r2=random.Random(s+7); steps=0
        while not obs.get("done") and steps<2000:
            aid=obs["metadata"]["current_agent_id"]
            p = a0_policy if (a0_policy and aid==0) else policy
            obs=env.step(p(aid,obs,rng=r2)); steps+=1
        st=env._episode; a0=st.agents[0]
        for a in st.agents:
            if not a.is_alive: causes[a.death_cause]+=1
        a0c[a0.death_cause or 'alive']+=1
        res.append(dict(T=st.step_count, alive=sum(a.is_alive for a in st.agents),
            healthy=sum(a.is_alive and a.infection_state=='none' for a in st.agents),
            a0_alive=a0.is_alive, a0_life=a0.death_step if a0.death_step is not None else st.step_count,
            a0_inf=st.agents[0].infection_role))
    out=dict(ep_len=S.mean(r['T'] for r in res), alive=S.mean(r['alive'] for r in res),
             healthy=S.mean(r['healthy'] for r in res), surv=sum(r['healthy']>=1 for r in res)/n,
             reached100=sum(r['T']>=100 for r in res)/n, a0_life=S.mean(r['a0_life'] for r in res))
    return out, causes, a0c, sorted(r['T'] for r in res)

if __name__=='__main__':
    t=time.time()
    o,c,a0c,lens=run(forage_heuristic_action)
    print(o); print(c); print('A0:',a0c); print(lens); print('sec',time.time()-t)
