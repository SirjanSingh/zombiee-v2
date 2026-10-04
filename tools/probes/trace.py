import sys; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..'))
from planner import *
env=SurviveCityV2Env(); obs=env.reset(seed=12345); P=Planner(env,thr=9); last=-1
while not obs.get("done"):
    ep=env._episode
    if ep.step_count!=last:
        last=ep.step_count
        print(f"t={last:2d} Z={[(z.row,z.col) for z in ep.zombies]} ", " | ".join(f"A{a.agent_id}{'*' if a.infection_state!='none' else ''}@{a.row},{a.col} hp{a.hp} h{a.hunger} t{a.thirst} {a.inventory}" for a in ep.agents if a.is_alive))
    aid=obs["metadata"]["current_agent_id"]; act=P(aid,obs); obs=env.step(act)
for a in env._episode.agents: print(a.agent_id,a.death_cause,a.death_step)
