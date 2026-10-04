import sys; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..'))
from planner2 import *
class Oracle(Camp):
    def __call__(self, aid, obs, rng=None):
        ep=self.env._episode; me=ep.agents[aid]; s=ep.step_count
        A=lambda t,**k: dict(agent_id=aid, action_type=t, **k)
        inf=[a for a in ep.agents if a.is_alive and a.infection_state!="none" and a.agent_id!=aid]
        if s in (30,50,70,90) and me.infection_state=="none":
            tg=[a.agent_id for a in inf if not a.locked_out]
            if tg: return A("vote_lockout", vote_target=tg[0])
        act=super().__call__(aid,obs,rng)
        # healthy agent: if a revealed biter is adjacent, step to a safehouse cell not adjacent to any infected
        if me.infection_state=="none" and act["action_type"]=="wait" and (me.row,me.col) in SAFEHOUSE_CELLS:
            bad={(a.row+dr,a.col+dc) for a in inf for dr,dc in [(0,0),(1,0),(-1,0),(0,1),(0,-1)]}
            if (me.row,me.col) in bad:
                for m,(dr,dc) in MOVES.items():
                    n=(me.row+dr,me.col+dc)
                    if n in SAFEHOUSE_CELLS and n not in bad: return A(m)
        return act
if __name__=="__main__":
    for kw in [dict(),dict(stock=2),dict(zclear=6)]:
        print("oracle",kw,*run2(cls=Oracle,**kw))
