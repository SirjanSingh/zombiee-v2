"""Scripted policies for rollouts, baselines and scenario generation.

Every policy has the signature `policy(agent_id, obs, rng=None) -> action dict`
and reads only the public observation (own stats/inventory, positions, zombies,
metadata.depleted_food, metadata.bite_history). No env internals, so the same
policy can drive training rollouts, eval opponents and SFT data.

    camp          water-camp planner (port of tools/probes/planner2.py Camp):
                  stock water early, camp in the safehouse, sortie only when the
                  nearest water is clear of zombies, vote out agents seen biting.
    heuristic_v3  training.inference.forage_heuristic_v3 (simple forager)
    random        training.inference.random_action
    mix           per-agent behaviour mix for scenario generation (see MixPolicy)
"""

from __future__ import annotations

import random
from collections import deque
from typing import Callable, Optional

from survivecity_v2_env.layout import (
    FOOD_CELLS, GRID_COLS, GRID_ROWS, SAFEHOUSE_CELLS, WALL_CELLS, WATER_CELLS,
)
from training.inference import forage_heuristic_v3, random_action

Policy = Callable[..., dict]

MOVES = {"move_up": (-1, 0), "move_down": (1, 0), "move_left": (0, -1), "move_right": (0, 1)}
VOTE_STEPS = (30, 50, 70, 90)


def _walk(r: int, c: int) -> bool:
    return 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS and (r, c) not in WALL_CELLS


def _mdist(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def bfs(start, goals, danger):
    """(distance, first move) to the nearest goal, avoiding danger cells unless they are goals."""
    if start in goals:
        return 0, None
    q = deque([(start, None, 0)])
    seen = {start}
    while q:
        (r, c), first, d = q.popleft()
        for m, (dr, dc) in MOVES.items():
            n = (r + dr, c + dc)
            if n in seen or not _walk(*n):
                continue
            if n in danger and n not in goals:
                continue
            seen.add(n)
            f = first or m
            if n in goals:
                return d + 1, f
            q.append((n, f, d + 1))
    return None, None


# ---------------------------------------------------------------------------
# Extraction run (W4): shared by camp_action, tools/probes Camp and Oracle
# ---------------------------------------------------------------------------

EXTRACT_SLACK = 6        # spare turns budgeted on top of the path length (zombie detours, waits)
EXTRACT_STOCK = 2        # water to carry before leaving for the zone


def extraction_action(pos, step: int, zone, extraction_step: int, zs, inv, hunger: int, thirst: int,
                      food, drink_at: int = 13, slack: int = EXTRACT_SLACK,
                      avoid=()) -> Optional[tuple]:
    """Next move of a timed extraction run, or None while it is not yet time to leave.

    Leaves the safehouse when the turns left before the helicopter drop to
    path length + `slack`, eats/drinks on the way (starvation is lethal outside
    the safehouse), avoids cells next to zombies, then holds inside the zone,
    sidestepping zombies. `avoid` = extra cells to keep off (e.g. next to a
    known infected agent). Returns (action_type, kwargs).
    """
    if not zone:
        return None
    zone = set(map(tuple, zone))
    turns_left = extraction_step - step
    danger = {(zr + dr, zc + dc) for zr, zc in zs for dr in (-1, 0, 1) for dc in (-1, 0, 1)
              if abs(dr) + abs(dc) <= 1} | set(avoid)
    zmin = min((_mdist(pos, z) for z in zs), default=99)
    nw = inv.count("water")

    if pos in zone:
        if thirst >= drink_at and nw:
            return "drink", {}
        if pos in food and hunger >= 4 and zmin > 1:
            return "eat", {}
        if hunger >= 12 and "food" in inv:
            return "eat", {}
        if pos in danger:
            for m, (dr, dc) in MOVES.items():
                n = (pos[0] + dr, pos[1] + dc)
                if n in zone and n not in danger and _walk(*n):
                    return m, {}
        return "wait", {}

    d_est, _ = bfs(pos, zone, set())
    if d_est is None:
        return None
    if turns_left > d_est + slack:
        return None

    # On the way: refuel when standing on a supply, detour for food/water when low.
    if pos in WATER_CELLS and thirst >= 3 and zmin > 1:
        return "drink", {}
    if pos in food and hunger >= 4 and zmin > 1:
        return "eat", {}
    if pos in WATER_CELLS and nw < EXTRACT_STOCK and len(inv) < 3 and zmin > 2:
        return "pickup", {"item_type": "water"}
    if hunger >= 9 and "food" not in inv and food:
        d, m = bfs(pos, set(food), danger)
        if m and d <= 3 and d + d_est <= turns_left:
            return m, {}
    if thirst >= 9 and nw == 0:
        d, m = bfs(pos, set(WATER_CELLS), danger)
        if m and d <= 3 and d + d_est <= turns_left:
            return m, {}
    if thirst >= drink_at and nw:
        return "drink", {}
    if hunger >= 12 and "food" in inv:
        return "eat", {}
    d, m = bfs(pos, zone, danger)
    if m:
        return m, {}
    return "wait", {}


FED_EAT_AT = 12          # eat carried food before starvation damage starts (15)
FED_SORTIE_MAX = 12      # only leave the safehouse for food while not starving


def keep_fed_action(pos, inv, hunger: int, zs, food, insafe: bool, zclear: int = 4) -> Optional[tuple]:
    """Hunger management for the extraction game, or None.

    The plain camp planner never eats: inside the safehouse healing cancels
    starvation damage, so it hides at hunger >= 15 forever. That makes any
    late trip outside lethal (-1 HP per turn), so with extraction on the camp
    planner carries one food item and eats it before starvation starts.
    """
    zmin = min((_mdist(pos, z) for z in zs), default=99)
    danger = {(zr + dr, zc + dc) for zr, zc in zs for dr in (-1, 0, 1) for dc in (-1, 0, 1)
              if abs(dr) + abs(dc) <= 1}
    has_food = "food" in inv
    if has_food and hunger >= FED_EAT_AT:
        return "eat", {}
    if insafe:
        if not has_food and len(inv) < 3 and hunger < FED_SORTIE_MAX:
            tgt = [f for f in food if min((_mdist(f, z) for z in zs), default=99) > zclear]
            if tgt:
                d, m = bfs(pos, set(tgt), danger)
                if m and d <= 4:
                    return m, {}
        return None
    if pos in food and zmin > 1:
        if hunger >= 4:
            return "eat", {}
        if not has_food and len(inv) < 3:
            return "pickup", {"item_type": "food"}
    if not has_food and len(inv) < 3 and zmin > 2:
        d, m = bfs(pos, set(food), danger)
        if m and d <= 3:
            return m, {}
    return None


def _me(obs: dict, agent_id: int) -> Optional[dict]:
    return next((a for a in obs.get("agents", [])
                 if a.get("agent_id") == agent_id and a.get("is_alive", True)), None)


def camp_action(agent_id: int, obs: dict, rng=None, drink_at: int = 13, stock: int = 3,
                zclear: int = 4, vote: bool = True) -> dict:
    """Water-camp planner on public observations. Parameters match planner2.Camp defaults."""
    me = _me(obs, agent_id)
    A = lambda t, **k: {"agent_id": agent_id, "action_type": t, **k}  # noqa: E731
    if me is None:
        return A("wait")
    meta = obs.get("metadata", {}) or {}
    pos = (me["row"], me["col"])
    s = obs.get("step_count", 0)
    inv = me.get("inventory", []) or []
    hunger, thirst = me.get("hunger", 0), me.get("thirst", 0)
    zs = [(z["row"], z["col"]) for z in obs.get("zombies", [])]
    zmin = min((_mdist(pos, z) for z in zs), default=99)
    danger = {(zr + dr, zc + dc) for zr, zc in zs for dr in (-1, 0, 1) for dc in (-1, 0, 1)
              if abs(dr) + abs(dc) <= 1}
    depleted = {tuple(c) for c in meta.get("depleted_food", [])}
    food = {c for c in FOOD_CELLS if c not in depleted}
    insafe = pos in SAFEHOUSE_CELLS
    nw = inv.count("water")

    if s in VOTE_STEPS:
        alive = {a["agent_id"] for a in obs.get("agents", []) if a.get("is_alive", True)}
        biters = [b["biter_id"] for b in meta.get("bite_history", [])
                  if b["biter_id"] in alive and b["biter_id"] != agent_id]
        if vote and biters:
            return A("vote_lockout", vote_target=biters[0])
    zone = meta.get("extraction_zone") or []
    if zone:
        ext = extraction_action(pos, s, zone, meta["extraction"]["extraction_step"], zs, inv,
                                hunger, thirst, food, drink_at=drink_at)
        if ext is not None:
            return A(ext[0], **ext[1])
    if "extraction" in meta:          # extraction game: stay fed, leave a slot for food
        stock = min(stock, EXTRACT_STOCK)
        fed = keep_fed_action(pos, inv, hunger, zs, food, insafe, zclear)
        if fed is not None:
            return A(fed[0], **fed[1])
    if insafe:
        if thirst >= drink_at and nw:
            return A("drink")
        if thirst >= 14 and hunger >= 14 and "food" in inv:
            return A("eat")
        if nw < (1 if s > 10 else stock) + (EXTRACT_STOCK - 1 if zone else 0) and s < 92:
            tgt = [w for w in WATER_CELLS if min((_mdist(w, z) for z in zs), default=99) > zclear]
            if tgt:
                d, m = bfs(pos, set(tgt), danger)
                if m and d <= 4:
                    return A(m)
        return A("wait")
    if pos in WATER_CELLS:
        if thirst >= 1 and not zmin <= 1:
            return A("drink") if thirst >= 3 or nw >= stock else A("pickup", item_type="water")
        if nw < stock and len(inv) < 3 and zmin > 2:
            return A("pickup", item_type="water")
    if pos in food and hunger >= 4 and zmin > 1:
        return A("eat")
    if nw < stock and len(inv) < 3 and zmin > 2:
        d, m = bfs(pos, set(WATER_CELLS), danger)
        if m and d <= 3:
            return A(m)
    d, m = bfs(pos, SAFEHOUSE_CELLS, danger)
    if m is None:
        d, m = bfs(pos, SAFEHOUSE_CELLS, set())
    return A(m) if m else A("wait")


def heuristic_v3_action(agent_id: int, obs: dict, rng=None) -> dict:
    return forage_heuristic_v3(agent_id, obs, rng=rng)


def random_policy(agent_id: int, obs: dict, rng=None) -> dict:
    return random_action(agent_id, obs, rng=rng)


POLICIES: dict[str, Policy] = {
    "camp": camp_action,
    "heuristic_v3": heuristic_v3_action,
    "random": random_policy,
}


def get_policy(name: str) -> Policy:
    try:
        return POLICIES[name]
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; have {sorted(POLICIES)}")


# ---------------------------------------------------------------------------
# Behaviour mix for scenario generation
# ---------------------------------------------------------------------------

# One letter per agent: c = camp, h = heuristic_v3, e = epsilon-camp (30% random).
MIX_CODES = {"c": "camp", "h": "heuristic_v3", "e": "eps_camp"}
EPS_RANDOM = 0.3


def sample_mix(rng: random.Random, n_agents: int = 5) -> str:
    """Seeded per-agent behaviour mix, e.g. 'chech'. Weighted toward the planner."""
    return "".join(rng.choices("che", weights=(0.5, 0.3, 0.2), k=n_agents))


class MixPolicy:
    """Each agent follows its own letter of `mix`. Deterministic given (mix, seed)."""

    def __init__(self, mix: str, seed):
        self.mix = mix
        self.rng = random.Random(f"mix|{seed}|{mix}")

    def __call__(self, agent_id: int, obs: dict, rng=None) -> dict:
        code = self.mix[agent_id] if agent_id < len(self.mix) else "c"
        if code == "h":
            return heuristic_v3_action(agent_id, obs, rng=self.rng)
        if code == "e" and self.rng.random() < EPS_RANDOM:
            return random_action(agent_id, obs, rng=self.rng)
        return camp_action(agent_id, obs)
