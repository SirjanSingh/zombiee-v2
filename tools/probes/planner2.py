"""Water-camp planner: stock water early, camp in safehouse, re-supply when coast is clear."""
import sys; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..'))
from planner import *
from training.policies import extraction_action, keep_fed_action, EXTRACT_STOCK
def mdist(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])
class Camp:
    def __init__(self, env, drink_at=13, stock=3, zclear=4, vote=True):
        self.env=env; self.drink_at=drink_at; self.stock=stock; self.zclear=zclear; self.vote=vote
    def avoid_cells(self, aid): return ()
    def __call__(self, aid, obs, rng=None):
        ep=self.env._episode; me=ep.agents[aid]; A=lambda t,**k: dict(agent_id=aid, action_type=t, **k)
        pos=(me.row,me.col); s=ep.step_count; inv=me.inventory
        zs=[(z.row,z.col) for z in ep.zombies]; zmin=min((mdist(pos,z) for z in zs), default=99)
        danger={(zr+dr,zc+dc) for zr,zc in zs for dr in range(-1,2) for dc in range(-1,2) if abs(dr)+abs(dc)<=1}
        food={c for c in FOOD_CELLS if ep.food_present.get(c,True)}
        insafe=pos in SAFEHOUSE_CELLS; nw=inv.count("water")
        if s in (30,50,70,90):
            biters=[b["biter_id"] for b in ep.bite_history if ep.agents[b["biter_id"]].is_alive and b["biter_id"]!=aid]
            if self.vote and biters: return A("vote_lockout", vote_target=biters[0])
        # W4: after the radio, timed extraction run (shared with training/policies.camp_action)
        zone=ep.extraction_zone
        if zone:
            ext=extraction_action(pos,s,zone,ep.balance.extraction_step,zs,inv,me.hunger,me.thirst,food,
                                  drink_at=self.drink_at,avoid=self.avoid_cells(aid))
            if ext is not None: return A(ext[0],**ext[1])
        stock=self.stock
        if ep.balance.extraction_enabled:   # extraction game: stay fed, leave a slot for food
            stock=min(stock,EXTRACT_STOCK)
            fed=keep_fed_action(pos,inv,me.hunger,zs,food,insafe,self.zclear,me.thirst,self.drink_at)
            if fed is not None: return A(fed[0],**fed[1])
        if insafe:
            if me.thirst>=self.drink_at and nw: return A("drink")
            if me.thirst>=14 and me.hunger>=14 and "food" in inv: return A("eat")
            # sortie if low on water and nearest water reachable with no zombie near it
            if nw< (1 if s>10 else stock)+(EXTRACT_STOCK-1 if zone else 0) and s<92:
                d,m=bfs(pos,set(WATER_CELLS),danger)
                tgt=[w for w in WATER_CELLS if min((mdist(w,z) for z in zs),default=99)>self.zclear]
                if tgt:
                    d,m=bfs(pos,set(tgt),danger)
                    if m and d<=4: return A(m)
            return A("wait")
        # outside
        if pos in WATER_CELLS:
            if me.thirst>=1 and not (zmin<=1): return A("drink") if me.thirst>=3 or nw>=stock else A("pickup",item_type="water")
            if nw<stock and len(inv)<3 and zmin>2: return A("pickup",item_type="water")
        if pos in food and me.hunger>=4 and zmin>1: return A("eat")
        if nw<stock and len(inv)<3 and zmin>2:
            d,m=bfs(pos,set(WATER_CELLS),danger)
            if m and d<=3: return A(m)
        d,m=bfs(pos,SAFEHOUSE_CELLS,danger)
        if m is None: d,m=bfs(pos,SAFEHOUSE_CELLS,set())
        return A(m) if m else A("wait")
def run2(n=100, seed=42, cls=Camp, **kw):
    rng=random.Random(seed); res=[]; causes=collections.Counter()
    for _ in range(n):
        env=SurviveCityV2Env(); sd=rng.randint(0,999999); obs=env.reset(seed=sd); P=cls(env,**kw)
        while not obs.get("done"):
            aid=obs["metadata"]["current_agent_id"]; obs=env.step(P(aid,obs))
        st=env._episode
        for a in st.agents:
            if not a.is_alive: causes[a.death_cause]+=1
        res.append((st.step_count, sum(a.is_alive for a in st.agents), sum(a.is_alive and a.infection_state=='none' for a in st.agents), st.agents[0].death_step or st.step_count))
    return dict(ep_len=S.mean(r[0] for r in res), alive=S.mean(r[1] for r in res), healthy=S.mean(r[2] for r in res),
                surv=sum(r[2]>=1 for r in res)/n, reach100=sum(r[0]>=100 for r in res)/n, a0_life=S.mean(r[3] for r in res)), causes
if __name__=="__main__":
    for kw in [dict(),dict(drink_at=14),dict(zclear=6),dict(vote=False),dict(stock=2)]:
        print(kw,*run2(**kw))
