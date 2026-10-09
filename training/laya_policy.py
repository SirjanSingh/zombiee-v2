"""Laya (421M ModernBERT "System One" decision model) as the A0 policy for v3.

Laya never generates text: it reads a state and a typed `choice` question and returns
a probability per option in one forward pass. Each A0 turn is one such question over
the 10 actions (same action set as training/tiny_policy.py). Trained by DAgger:
round 0 = teacher rollouts (behaviour cloning), round r = Laya plays, teacher labels.

State text = the env's own per-turn description (what Qwen reads) + a compact map
section, because Laya's 1,024-token context can't hold Qwen's long system prompt
with the full layout. The map section only restates public information: nearest
water / food / safehouse / extraction zone with distances, and per move whether it
is blocked or ends next to a zombie.

  # build round-0 data (CPU, no model needed)
  python -m training.laya_policy rows --teacher camp --episodes 300 --out data/laya_r0.jsonl
  # DAgger round with a trained Laya student (GPU)
  python -m training.laya_policy rows --teacher camp --episodes 200 --student ckpt/laya_r0 --out data/laya_r1.jsonl
  # evaluate
  python -m training.laya_policy eval --student ckpt/laya_r1 --n-eval 200
Fine-tuning itself: laya.train.finetune(jsonl, base_ckpt, out_dir, TrainConfig(...)), see tools/laya_dagger.sh.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import time
from typing import Optional

from survivecity_v2_env.balance import get_balance
from survivecity_v2_env.env import SurviveCityV2Env
from training.policies import (FOOD_CELLS, SAFEHOUSE_CELLS, WATER_CELLS, MOVES, _walk, bfs, get_policy)
from training.tiny_policy import ACTIONS, action_label, legal_mask, to_action

QID = "action"
QUESTION = {QID: {
    "type": "choice",
    "instructions": "You are A0 in a zombie survival game. Pick A0's next action. Stay fed and watered "
                    "(hunger or thirst 15 is deadly outside the safehouse), avoid cells next to zombies, "
                    "and be inside the extraction zone when the helicopter loads.",
    "criteria": {
        "move_up": "step one cell up (row - 1)",
        "move_down": "step one cell down (row + 1)",
        "move_left": "step one cell left (col - 1)",
        "move_right": "step one cell right (col + 1)",
        "wait": "stay on this cell",
        "drink": "drink carried water or drink from the water cell you stand on",
        "eat": "eat carried food or eat from the food cell you stand on",
        "pickup_water": "pick up water from the water cell you stand on",
        "pickup_food": "pick up food from the food cell you stand on",
        "vote": "vote to lock out the teammate seen biting",
    },
}}
assert list(QUESTION[QID]["criteria"]) == ACTIONS


def _near(pos, cells, k=2):
    out = []
    for c in cells:
        d, _ = bfs(pos, {c}, set())
        if d is not None:
            out.append((d, c))
    return sorted(out)[:k]


def state_text(obs: dict) -> str:
    desc = obs.get("description", "")
    me = next((a for a in obs.get("agents", []) if a["agent_id"] == 0 and a.get("is_alive", True)), None)
    if me is None:
        return desc
    pos = (me["row"], me["col"])
    meta = obs.get("metadata", {}) or {}
    depleted = {tuple(c) for c in meta.get("depleted_food", [])}
    zs = [(z["row"], z["col"]) for z in obs.get("zombies", [])]
    danger = {(zr + dr, zc + dc) for zr, zc in zs for dr in (-1, 0, 1) for dc in (-1, 0, 1) if abs(dr) + abs(dc) <= 1}
    lines = ["MAP:"]
    lines.append("  on cell: " + ("safehouse" if pos in SAFEHOUSE_CELLS else "water" if pos in WATER_CELLS
                                  else "food" if pos in FOOD_CELLS and pos not in depleted else "open"))
    w = _near(pos, WATER_CELLS)
    f = _near(pos, [c for c in FOOD_CELLS if c not in depleted])
    lines.append("  nearest water: " + ", ".join(f"({r},{c}) {d} steps" for d, (r, c) in w))
    lines.append("  nearest food: " + (", ".join(f"({r},{c}) {d} steps" for d, (r, c) in f) or "none left"))
    ds, _ = bfs(pos, set(SAFEHOUSE_CELLS), set())
    lines.append(f"  safehouse: {ds} steps")
    zone = {tuple(c) for c in (meta.get("extraction_zone") or [])}
    if zone:
        dz, _ = bfs(pos, zone, set())
        lines.append(f"  extraction zone: {dz} steps" + (" (you are in it)" if pos in zone else ""))
    moves = []
    for m, (dr, dc) in MOVES.items():
        n = (pos[0] + dr, pos[1] + dc)
        if not _walk(*n):
            moves.append(f"{m} blocked")
        elif n in set(zs):
            moves.append(f"{m} zombie")
        elif n in danger:
            moves.append(f"{m} next-to-zombie")
        else:
            moves.append(f"{m} safe")
    lines.append("  moves: " + ", ".join(moves))
    return desc + "\n" + "\n".join(lines)


def row(obs: dict, label_idx: int) -> dict:
    return {"state": state_text(obs), "questions": QUESTION, "expected": {QID: ACTIONS[label_idx]}}


class LayaPolicy:
    """A0 policy backed by a Laya checkpoint; illegal actions are masked before the argmax."""

    def __init__(self, ckpt: str):
        import laya
        self.agent = laya.load(ckpt)
        self.calls = 0
        self.secs = 0.0

    def probs(self, obs: dict) -> list[float]:
        t = time.time()
        res = self.agent.predict(state_text(obs), QUESTION)
        self.secs += time.time() - t
        self.calls += 1
        ans = (res.get("answers") or res)[QID]
        return [ans["probabilities"].get(a, 0.0) for a in ACTIONS]

    def __call__(self, agent_id: int, obs: dict, rng=None) -> dict:
        p = self.probs(obs)
        mask = legal_mask(obs)
        best = max((i for i in range(len(ACTIONS)) if mask[i]), key=lambda i: p[i], default=ACTIONS.index("wait"))
        return to_action(best, obs)


def rollout(seed: int, a0_policy, teacher=None, teammate: str = "camp", balance: str = "v3-rc2"):
    env = SurviveCityV2Env(balance=get_balance(balance), a0_healthy=True)
    obs = env.reset(seed=seed)
    mate = get_policy(teammate)
    rng = random.Random(f"laya|{seed}")
    rows = []
    a0_log = []
    while not obs.get("done"):
        aid = obs["metadata"]["current_agent_id"]
        if aid == 0:
            if teacher is not None:
                lab = action_label(teacher(0, obs, rng=rng))
                if lab is not None:
                    rows.append(row(obs, lab))
            act = a0_policy(0, obs, rng=rng)
            a0_log.append(act)
        else:
            act = mate(aid, obs, rng=rng)
        obs = env.step(act)
    ep = env._episode
    a0 = ep.agents[0]
    res = ep.extraction_result or {}
    rec = {"seed": seed, "a0_ext": 0 in list(res.get("extracted", [])), "team": bool(res.get("success")),
           "a0_alive": a0.is_alive, "life": a0.death_step if a0.death_step is not None else ep.step_count,
           "cause": a0.death_cause if not a0.is_alive else "alive", "a0_actions": a0_log}
    return rec, rows


def summarize(recs: list[dict]) -> dict:
    n = len(recs)
    return {"n": n, "a0_extracted": sum(r["a0_ext"] for r in recs) / n, "team": sum(r["team"] for r in recs) / n,
            "a0_alive": sum(r["a0_alive"] for r in recs) / n, "a0_life": sum(r["life"] for r in recs) / n,
            "causes": dict(collections.Counter(r["cause"] for r in recs).most_common())}


def seeds(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(0, 999999) for _ in range(n)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["rows", "eval", "show", "record"])
    p.add_argument("--teacher", default="camp")
    p.add_argument("--student", default=None, help="Laya checkpoint; omit in `rows` to roll out the teacher")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seed", type=int, default=5150, help="rollout seeds (eval seeds are --eval-seed)")
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--eval-seed", type=int, default=999)
    p.add_argument("--out", default="data/laya_rows.jsonl")
    p.add_argument("--log", default=None)
    args = p.parse_args(argv)

    if args.cmd == "show":
        env = SurviveCityV2Env(balance=get_balance("v3-rc2"), a0_healthy=True)
        obs = env.reset(seed=58523)
        camp = get_policy("camp")
        r = random.Random(0)
        while obs["step_count"] < 64 or obs["metadata"]["current_agent_id"] != 0:
            obs = env.step(camp(obs["metadata"]["current_agent_id"], obs, rng=r))
        print(state_text(obs))
        return

    teacher = get_policy(args.teacher)
    student = LayaPolicy(args.student) if args.student else None
    if args.cmd == "rows":
        t = time.time()
        recs, n = [], 0
        with open(args.out, "w", encoding="utf-8") as f:
            for s in seeds(args.episodes, args.seed):
                rec, rows = rollout(s, student or teacher, teacher)
                recs.append(rec)
                for rw in rows:
                    f.write(json.dumps(rw) + "\n")
                n += len(rows)
        print(f"rows: {n} labelled states from {len(recs)} episodes ({'student ' + args.student if student else 'teacher'})"
              f" -> {args.out}; rollout policy {summarize(recs)} ({time.time() - t:.0f}s)")
        return

    if args.cmd == "record":     # Laya plays; A0 actions saved so CPU workers can replay + label exactly
        t = time.time()
        recs = []
        with open(args.out, "w", encoding="utf-8") as f:
            for s in seeds(args.episodes, args.seed):
                rec, _ = rollout(s, student)
                recs.append(rec)
                f.write(json.dumps(rec) + "\n")
        print(f"record: {len(recs)} episodes -> {args.out}; {summarize(recs)} ({time.time() - t:.0f}s)")
        return

    t = time.time()
    recs = [rollout(s, student)[0] for s in seeds(args.n_eval, args.eval_seed)]
    for r in recs:
        r.pop("a0_actions", None)
    out = summarize(recs)
    out.update({"student": args.student, "eval_seed": args.eval_seed,
                "ms_per_decision": round(1000 * student.secs / max(1, student.calls), 1)})
    print(json.dumps(out), f"({time.time() - t:.0f}s)")
    if args.log:
        with open(args.log, "a", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
