"""Episode replay recorder (W5): full ground-truth state after every action.

Used for review and as the footage source for the YouTube series. A replay is a
JSON file:

    {"meta":  {seed, balance, a0_policy, teammates, label, a0_role, ...},
     "layout": {rows, cols, walls, food, water, medicine, safehouse},
     "frames": [{t, actor, action, day, agents, zombies, food_down, events}, ...]}

`agents` rows are [id, row, col, hp, hunger, thirst, alive, infection, role,
locked_out, inventory]. Infection and role are the TRUE values (the viewer sees
the traitors; the agents themselves do not). `events` are diffs against the
previous frame: death, bite, reveal, lockout, eat, drink, pickup, wave.
Frame 0 is the reset state (actor = null).
"""

from __future__ import annotations

import json
import os
from typing import Optional

from survivecity_v2_env.layout import (
    FOOD_CELLS, GRID_COLS, GRID_ROWS, MEDICINE_CELLS, SAFEHOUSE_CELLS, WALL_CELLS, WATER_CELLS,
)

REPLAY_VERSION = 1


def _cells(cells) -> list[list[int]]:
    return sorted([r, c] for r, c in cells)


def layout_dict() -> dict:
    return {"rows": GRID_ROWS, "cols": GRID_COLS, "walls": _cells(WALL_CELLS),
            "food": _cells(FOOD_CELLS), "water": _cells(WATER_CELLS),
            "medicine": _cells(MEDICINE_CELLS), "safehouse": _cells(SAFEHOUSE_CELLS)}


def _compact_action(action: Optional[dict]) -> Optional[dict]:
    if action is None:
        return None
    return {k: v for k, v in action.items() if k != "agent_id" and v is not None}


class ReplayRecorder:
    """Call `capture(actor, action)` right after every env.step(action)."""

    def __init__(self, env, meta: Optional[dict] = None):
        self.env = env
        ep = env._episode
        self.meta = {"version": REPLAY_VERSION, "seed": ep.episode_seed,
                     "balance": ep.balance.to_dict(), "max_steps": ep.max_steps, **(meta or {})}
        a0 = ep.agents[0]
        self.meta["a0_role"] = a0.infection_role if a0.infection_state != "none" else "healthy"
        self.frames: list[dict] = []
        self._prev: Optional[dict] = None
        self._bites_seen = 0
        self._zombies_seen = len(ep.zombies)
        self.capture(None, None)

    def _snapshot(self) -> dict:
        ep = self.env._episode
        return {
            "agents": [[a.agent_id, a.row, a.col, a.hp, a.hunger, a.thirst, a.is_alive,
                        a.infection_state, a.infection_role, a.locked_out, list(a.inventory)]
                       for a in ep.agents],
            "zombies": [[z.row, z.col] for z in ep.zombies],
            "food_down": sorted([r, c] for (r, c), present in ep.food_present.items() if not present),
        }

    def capture(self, actor: Optional[int], action: Optional[dict]) -> None:
        ep = self.env._episode
        snap = self._snapshot()
        events: list[dict] = []
        prev = self._prev
        if prev is not None:
            for a_now, a_prev in zip(snap["agents"], prev["agents"]):
                aid = a_now[0]
                if a_prev[6] and not a_now[6]:
                    events.append({"type": "death", "agent": aid,
                                   "cause": ep.agents[aid].death_cause})
                if a_prev[7] == "latent" and a_now[7] == "revealed":
                    events.append({"type": "reveal", "agent": aid, "role": a_now[8]})
                if not a_prev[9] and a_now[9]:
                    events.append({"type": "lockout", "agent": aid})
        for b in ep.bite_history[self._bites_seen:]:
            events.append({"type": "bite", "biter": b["biter_id"], "victim": b["victim_id"]})
        self._bites_seen = len(ep.bite_history)
        if len(ep.zombies) > self._zombies_seen:
            events.append({"type": "wave", "new": len(ep.zombies) - self._zombies_seen})
        self._zombies_seen = len(ep.zombies)
        if actor is not None and action is not None:
            ag = ep.agents[actor]
            if ag.ate_this_step:
                events.append({"type": "eat", "agent": actor})
            if ag.drank_this_step:
                events.append({"type": "drink", "agent": actor})
        self.frames.append({"t": ep.step_count, "actor": actor, "action": _compact_action(action),
                            "day": ep.day_phase, **snap, "events": events})
        self._prev = snap

    def to_dict(self) -> dict:
        ep = self.env._episode
        a0 = ep.agents[0]
        self.meta["result"] = {
            "final_t": ep.step_count,
            "a0_alive": a0.is_alive,
            "a0_death_step": a0.death_step,
            "a0_death_cause": a0.death_cause,
            "healthy_alive": sum(1 for a in ep.agents if a.is_alive and a.infection_state == "none"),
        }
        return {"meta": self.meta, "layout": layout_dict(), "frames": self.frames}

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, separators=(",", ":"))
        return path
