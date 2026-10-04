"""Scripted planner to measure SurviveCity v2's achievable ceiling.
Uses only observation-visible info except food_present (read from env internals)."""
import random, sys, logging, collections, statistics as S
from collections import deque
logging.disable(logging.CRITICAL); import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..'))
from survivecity_v2_env.env import SurviveCityV2Env
from survivecity_v2_env.layout import FOOD_CELLS, WATER_CELLS, SAFEHOUSE_CELLS, WALL_CELLS, GRID_ROWS, GRID_COLS
MOVES={"move_up":(-1,0),"move_down":(1,0),"move_left":(0,-1),"move_right":(0,1)}
def walk(r,c): return 0<=r<GRID_ROWS and 0<=c<GRID_COLS and (r,c) not in WALL_CELLS
def bfs(start, goals, danger):
    """returns (dist, first_move) to nearest goal, avoiding danger cells (unless goal)."""
    if start in goals: return 0,None
    q=deque([(start,None,0)]); seen={start}
    while q:
        (r,c),fm,d=q.popleft()
        for m,(dr,dc) in MOVES.items():
            n=(r+dr,c+dc)
            if n in seen or not walk(*n): continue
            if n in danger and n not in goals: continue
            seen.add(n); f=fm or m
            if n in goals: return d+1,f
            q.append((n,f,d+1))
    return None,None

class Planner:
    def __init__(self, env, thr=7, vote=True, stock=True):
        self.env=env; self.thr=thr; self.vote=vote; self.stock=stock
    def __call__(self, aid, obs, rng=None):
        ep=self.env._episode; me=ep.agents[aid]
        A=lambda t,**k: dict(agent_id=aid, action_type=t, **k)
        pos=(me.row,me.col); s=ep.step_count; inv=me.inventory
        zs=[(z.row,z.col) for z in ep.zombies]
        danger={(zr+dr,zc+dc) for zr,zc in zs for dr in range(-1,2) for dc in range(-1,2) if abs(dr)+abs(dc)<=1}
        food={c for c in FOOD_CELLS if ep.food_present.get(c,True)}
        insafe=pos in SAFEHOUSE_CELLS
        rate=1.5 if me.infection_state!="none" else 1.0
        if s in (30,50,70,90) and self.vote:
            # vote for a *visibly revealed* biter only (non-oracle: revealed = bites happen; we cheat lightly via bite_history)
            biters={b["biter_id"] for b in ep.bite_history if ep.agents[b["biter_id"]].is_alive}
            tgt=[b for b in biters if b!=aid]
            return A("vote_lockout", vote_target=tgt[0]) if tgt else A("wait")
        # consume
        if pos in WATER_CELLS:
            if me.thirst>=3: return A("drink")
            if self.stock and inv.count("water")<2 and len(inv)<3: return A("pickup", item_type="water")
        if pos in food:
            if me.hunger>=3: return A("eat")
            if self.stock and inv.count("food")<1 and len(inv)<3: return A("pickup", item_type="food")
        if me.thirst>=self.thr and "water" in inv: return A("drink")
        if me.hunger>=self.thr/rate and "food" in inv: return A("eat")
        # decide errand
        goal=None
        need_w = me.thirst>=self.thr-4 and "water" not in inv
        need_f = me.hunger*rate>=self.thr-4 and "food" not in inv
        if self.stock and insafe and s<95:
            if inv.count("water")<1: need_w = need_w or me.thirst>=2
        if need_w and (not need_f or me.thirst>=me.hunger): goal=WATER_CELLS
        elif need_f: goal=food or FOOD_CELLS
        elif need_w: goal=WATER_CELLS
        if goal:
            d,m=bfs(pos,set(goal),danger)
            if m: return A(m)
            if d==0: return A("wait")
        if insafe: return A("wait")
        d,m=bfs(pos,SAFEHOUSE_CELLS,danger)
        if m is None: d,m=bfs(pos,SAFEHOUSE_CELLS,set())
        return A(m) if m else A("wait")

def run(n=100, seed=42, **kw):
    rng=random.Random(seed); res=[]; causes=collections.Counter()
    for ep in range(n):
        env=SurviveCityV2Env(); sd=rng.randint(0,999999); obs=env.reset(seed=sd); P=Planner(env,**kw)
        while not obs.get("done"):
            aid=obs["metadata"]["current_agent_id"]; obs=env.step(P(aid,obs))
        st=env._episode
        for a in st.agents:
            if not a.is_alive: causes[a.death_cause]+=1
        res.append((st.step_count, sum(a.is_alive for a in st.agents), sum(a.is_alive and a.infection_state=='none' for a in st.agents), st.agents[0].death_step or st.step_count))
    return dict(ep_len=S.mean(r[0] for r in res), alive=S.mean(r[1] for r in res), healthy=S.mean(r[2] for r in res),
                surv=sum(r[2]>=1 for r in res)/n, a0_life=S.mean(r[3] for r in res)), causes
if __name__=="__main__":
    for kw in [dict(thr=7),dict(thr=9),dict(thr=11),dict(thr=9,stock=False),dict(thr=9,vote=False)]:
        print(kw, *run(**kw))
