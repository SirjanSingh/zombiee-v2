import sys, logging; logging.disable(50)
sys.path.insert(0, r"D:\projs\extra\v2\zombiee-v2\tools"); sys.path.insert(0, r"D:\tmp\claude\D--projs-extra-v2-zombiee-v2\221339af-29e8-426b-86c1-3f2ee8ec4dce\scratchpad")
import calibrate as C
exec(open(r"D:\tmp\claude\D--projs-extra-v2-zombiee-v2\221339af-29e8-426b-86c1-3f2ee8ec4dce\scratchpad\heur_wall.py").read().split("base = ")[0])
def run(ov, n=100):
    cfg = C.get_balance().with_(starting_infected_progression=False, **ov)
    r = C.calibrate(cfg, ["random", "heuristic", "heur_wall", "camp", "oracle"], n=n, seed=42, progress=False)
    return {p: v["metrics"] for p, v in r.items()}

def heurFixed_factory(env, seed):
    import random as _r
    rng = _r.Random(seed + 7)
    def act(aid, obs):
        me = next((a for a in obs["agents"] if a["agent_id"] == aid), None)
        if me and (me["row"], me["col"]) in I._WATER_CELLS_TUPLE and me["thirst"] == 0:
            # bug 1 fix: don't drink when not thirsty; fall through to the rest of the rules
            pass
        ep = env._episode
        live = tuple(c for c in I._FOOD_CELLS_TUPLE if ep.food_present.get(c, True))
        oldF, oldW, oldS = I._FOOD_CELLS_TUPLE, I._WATER_CELLS_TUPLE, I._step_toward
        I._FOOD_CELLS_TUPLE = live or oldF
        if me and me["thirst"] == 0:
            I._WATER_CELLS_TUPLE = tuple(c for c in oldW if c != (me["row"], me["col"]))
        I._step_toward = step_wallaware
        try:
            return I.forage_heuristic_action(aid, obs, rng=rng)
        finally:
            I._FOOD_CELLS_TUPLE, I._WATER_CELLS_TUPLE, I._step_toward = oldF, oldW, oldS
    return act
C.POLICIES["heur_fixed"] = heurFixed_factory

def run2(ov, n=100, pols=("random", "heuristic", "heur_fixed", "camp", "oracle")):
    cfg = C.get_balance().with_(starting_infected_progression=False, **ov)
    r = C.calibrate(cfg, list(pols), n=n, seed=42, progress=False)
    return {p: (v["metrics"], v["healthy_death_causes"]) for p, v in r.items()}

def heurAvoid_factory(env, seed):
    base = heurFixed_factory(env, seed)
    from survivecity_v2_env.layout import SAFEHOUSE_CELLS
    def act(aid, obs):
        a = base(aid, obs)
        m = a.get("action_type")
        if m not in D: return a
        me = next(x for x in obs["agents"] if x["agent_id"] == aid)
        zs = {(z["row"], z["col"]) for z in obs["zombies"]}
        danger = {(r + dr, c + dc) for r, c in zs for dr, dc in [(0,0),(1,0),(-1,0),(0,1),(0,-1)]}
        nr, nc = me["row"] + D[m][0], me["col"] + D[m][1]
        if (nr, nc) not in danger or (nr, nc) in SAFEHOUSE_CELLS: return a
        here_safe = (me["row"], me["col"]) not in danger
        for alt in D:
            ar, ac = me["row"] + D[alt][0], me["col"] + D[alt][1]
            if ok(ar, ac) and ((ar, ac) not in danger or (ar, ac) in SAFEHOUSE_CELLS) and alt != m:
                if here_safe: return {"agent_id": aid, "action_type": "wait"}
                return {"agent_id": aid, "action_type": alt}
        return a
    return act
C.POLICIES["heur_avoid"] = heurAvoid_factory
