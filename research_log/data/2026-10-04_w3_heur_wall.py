import sys, logging, random; logging.disable(50)
sys.path.insert(0, r"D:\projs\extra\v2\zombiee-v2\tools")
import calibrate as C
import training.inference as I
from survivecity_v2_env.layout import WALL_CELLS, GRID_ROWS, GRID_COLS
D = {"move_up": (-1, 0), "move_down": (1, 0), "move_left": (0, -1), "move_right": (0, 1)}
def ok(r, c): return 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS and (r, c) not in WALL_CELLS
def step_wallaware(my_r, my_c, target):
    tr, tc = target; dr, dc = tr - my_r, tc - my_c
    v = ("move_down" if dr > 0 else "move_up") if dr else None
    h = ("move_right" if dc > 0 else "move_left") if dc else None
    order = [v, h] if abs(dr) > abs(dc) else [h, v]
    for m in order:
        if m and ok(my_r + D[m][0], my_c + D[m][1]): return m
    return next((m for m in order if m), None)
orig = I._step_toward
def heurW(aid, obs, rng=None):
    I._step_toward = step_wallaware
    try: return I.forage_heuristic_action(aid, obs, rng=rng)
    finally: I._step_toward = orig
C.POLICIES["heur_wall"] = C._rng_policy(heurW)
base = C.get_balance().with_(starting_infected_progression=False)
grid = [("v2.2+exempt", {}), ("rate .75", dict(hunger_rate=.75, thirst_rate=.75)),
        ("rate .6", dict(hunger_rate=.6, thirst_rate=.6)),
        ("rate .6 me2", dict(hunger_rate=.6, thirst_rate=.6, zombie_move_every=2)),
        ("rate .5", dict(hunger_rate=.5, thirst_rate=.5)),
        ("rate .5 me2", dict(hunger_rate=.5, thirst_rate=.5, zombie_move_every=2)),
        ("rate .5 me2 r5", dict(hunger_rate=.5, thirst_rate=.5, zombie_move_every=2, zombie_chase_radius=5))]
for name, ov in grid:
    r = C.calibrate(base.with_(**ov), ["heur_wall"], n=100, seed=42, progress=False)["heur_wall"]
    print(f"{name:<16}", r["metrics"]["survival"], r["metrics"]["ep_len"], r["metrics"]["a0_life"], r["metrics"]["healthy_end"], dict(list(r["healthy_death_causes"].items())[:3]))
