"""Tiny-network A0 policy for v3: the "does it need to be a 3B LLM?" baseline.

A small CNN reads the public observation as a 15x15 multi-channel grid plus a few
scalars and picks one of 10 actions, every A0 turn (no K-action plans). Trained
with DAgger against the same teachers as the Qwen runs, on CPU, in minutes.

  python -m training.tiny_policy --teacher camp --rounds 6 --eps-per-round 200 --out checkpoints/tiny_camp.pt
  python -m training.tiny_policy --eval checkpoints/tiny_camp.pt --n-eval 200

Eval seeds: random.Random(999) (same as tools/probes sweeps) and optionally the
eval_v3 final seeds (--eval-seed 4321 --n-eval 60) for the Qwen tables.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from survivecity_v2_env.balance import get_balance
from survivecity_v2_env.env import SurviveCityV2Env
from training.policies import (FOOD_CELLS, GRID_COLS, GRID_ROWS, SAFEHOUSE_CELLS, WALL_CELLS, WATER_CELLS,
                               get_policy)

ACTIONS = ["move_up", "move_down", "move_left", "move_right", "wait", "drink", "eat",
           "pickup_water", "pickup_food", "vote"]
A_IDX = {a: i for i, a in enumerate(ACTIONS)}
N_CH = 11
N_SCALAR = 12


def action_label(act: dict) -> Optional[int]:
    t = act.get("action_type")
    if t == "pickup":
        t = "pickup_" + act.get("item_type", "water")
    elif t == "vote_lockout":
        t = "vote"
    return A_IDX.get(t)


def to_action(idx: int, obs: dict) -> dict:
    a = ACTIONS[idx]
    if a.startswith("pickup_"):
        return {"agent_id": 0, "action_type": "pickup", "item_type": a.split("_")[1]}
    if a == "vote":
        meta = obs.get("metadata", {}) or {}
        alive = {x["agent_id"] for x in obs.get("agents", []) if x.get("is_alive", True)}
        biters = [b["biter_id"] for b in meta.get("bite_history", []) if b["biter_id"] in alive and b["biter_id"] != 0]
        if not biters:
            return {"agent_id": 0, "action_type": "wait"}
        return {"agent_id": 0, "action_type": "vote_lockout", "vote_target": biters[0]}
    return {"agent_id": 0, "action_type": a}


def featurize(obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Public observation of A0 -> (grid [N_CH,15,15], scalars [N_SCALAR])."""
    g = np.zeros((N_CH, GRID_ROWS, GRID_COLS), dtype=np.float32)
    g[0], g[1], g[2], g[3] = _STATIC
    meta = obs.get("metadata", {}) or {}
    depleted = {tuple(c) for c in meta.get("depleted_food", [])}
    for r, c in depleted:
        g[1, r, c] = 0
    for z in obs.get("zombies", []):
        g[4, z["row"], z["col"]] += 1
    for r, c in (meta.get("extraction_zone") or []):
        g[5, r, c] = 1
    alive = {a["agent_id"] for a in obs.get("agents", []) if a.get("is_alive", True)}
    biters = {b["biter_id"] for b in meta.get("bite_history", []) if b["biter_id"] in alive}
    me = None
    for a in obs.get("agents", []):
        if not a.get("is_alive", True):
            continue
        if a["agent_id"] == 0:
            me = a
            g[6, a["row"], a["col"]] = 1
        else:
            g[7, a["row"], a["col"]] += 1
            if a["agent_id"] in biters:
                g[8, a["row"], a["col"]] = 1
    # zombie danger halo (cells within 1 of a zombie) and distance-to-A0 ramp
    zs = [(z["row"], z["col"]) for z in obs.get("zombies", [])]
    for zr, zc in zs:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if abs(dr) + abs(dc) <= 1 and 0 <= zr + dr < GRID_ROWS and 0 <= zc + dc < GRID_COLS:
                    g[9, zr + dr, zc + dc] = 1
    s = np.zeros(N_SCALAR, dtype=np.float32)
    step = obs.get("step_count", 0)
    if me is not None:
        pr, pc = me["row"], me["col"]
        g[10] = 1.0 - (np.abs(_RR - pr) + np.abs(_CC - pc)) / 28.0
        inv = me.get("inventory", []) or []
        s[0] = me.get("hp", 0) / 3
        s[1] = min(me.get("hunger", 0), 30) / 15
        s[2] = min(me.get("thirst", 0), 30) / 15
        s[3] = inv.count("water") / 3
        s[4] = inv.count("food") / 3
        s[5] = inv.count("medicine") / 3
        s[6] = len(inv) / 3
        s[7] = min((abs(pr - z[0]) + abs(pc - z[1]) for z in zs), default=28) / 28
    s[8] = step / 100
    ext = meta.get("extraction") or {}
    if ext.get("extraction_step"):
        s[9] = max(0, ext["extraction_step"] - step) / 100
    s[10] = 1.0 if meta.get("extraction_zone") else 0.0
    s[11] = 1.0 if step in (30, 50, 70, 90) else 0.0
    return torch.from_numpy(g), torch.from_numpy(s)


def _static_planes() -> np.ndarray:
    st = np.zeros((4, GRID_ROWS, GRID_COLS), dtype=np.float32)
    for k, cells in enumerate((WALL_CELLS, FOOD_CELLS, WATER_CELLS, SAFEHOUSE_CELLS)):
        for r, c in cells:
            st[k, r, c] = 1
    return st


_STATIC = _static_planes()
_RR = np.arange(GRID_ROWS, dtype=np.float32).reshape(-1, 1)
_CC = np.arange(GRID_COLS, dtype=np.float32).reshape(1, -1)


class TinyNet(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(N_CH, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1, stride=2), nn.ReLU(),   # 15 -> 8
        )
        self.head = nn.Sequential(
            nn.Linear(width * 8 * 8 + N_SCALAR, 256), nn.ReLU(),
            nn.Linear(256, len(ACTIONS)),
        )

    def forward(self, g, s):
        return self.head(torch.cat([self.conv(g).flatten(1), s], 1))


def legal_mask(obs: dict) -> torch.Tensor:
    """Mask actions the env would treat as no-ops (walls, eating with nothing, ...)."""
    m = torch.ones(len(ACTIONS), dtype=torch.bool)
    me = next((a for a in obs.get("agents", []) if a["agent_id"] == 0 and a.get("is_alive", True)), None)
    if me is None:
        return m
    pos = (me["row"], me["col"])
    inv = me.get("inventory", []) or []
    for i, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
        r, c = pos[0] + dr, pos[1] + dc
        if not (0 <= r < GRID_ROWS and 0 <= c < GRID_COLS) or (r, c) in WALL_CELLS:
            m[i] = False
    meta = obs.get("metadata", {}) or {}
    depleted = {tuple(c) for c in meta.get("depleted_food", [])}
    on_food = pos in FOOD_CELLS and pos not in depleted
    if "water" not in inv and pos not in WATER_CELLS:
        m[A_IDX["drink"]] = False
    if "food" not in inv and not on_food:
        m[A_IDX["eat"]] = False
    if pos not in WATER_CELLS or len(inv) >= 3:
        m[A_IDX["pickup_water"]] = False
    if not on_food or len(inv) >= 3:
        m[A_IDX["pickup_food"]] = False
    if obs.get("step_count", 0) not in (30, 50, 70, 90):
        m[A_IDX["vote"]] = False
    return m


def net_policy(net: TinyNet, greedy: bool = True, temp: float = 1.0):
    def act(agent_id: int, obs: dict, rng=None) -> dict:
        g, s = featurize(obs)
        with torch.no_grad():
            logits = net(g[None], s[None])[0]
        logits[~legal_mask(obs)] = -1e9
        if greedy:
            i = int(logits.argmax())
        else:
            i = int(torch.distributions.Categorical(logits=logits / temp).sample())
        return to_action(i, obs)
    return act


def rollout(seed: int, a0_policy, teacher=None, balance="v3-rc2", teammate="camp",
            beta: float = 0.0, rng_key: str = "") -> tuple[dict, list]:
    """Play one episode; A0 = a0_policy (or the teacher with prob beta). Returns (record, labelled states)."""
    env = SurviveCityV2Env(balance=get_balance(balance), a0_healthy=True)
    obs = env.reset(seed=seed)
    mate = get_policy(teammate)
    rng = random.Random(f"tiny|{seed}|{rng_key}")
    data = []
    while not obs.get("done"):
        aid = obs["metadata"]["current_agent_id"]
        if aid == 0:
            label = None
            if teacher is not None:
                t_act = teacher(0, obs, rng=rng)
                label = action_label(t_act)
                if label is not None:
                    g, s = featurize(obs)
                    data.append((g, s, label))
            act = t_act if (teacher is not None and rng.random() < beta) else a0_policy(0, obs, rng=rng)
        else:
            act = mate(aid, obs, rng=rng)
        obs = env.step(act)
    ep = env._episode
    a0 = ep.agents[0]
    res = ep.extraction_result or {}
    rec = {"seed": seed, "a0_ext": 0 in list(res.get("extracted", [])), "team": bool(res.get("success")),
           "a0_alive": a0.is_alive, "life": a0.death_step if a0.death_step is not None else ep.step_count,
           "cause": a0.death_cause if not a0.is_alive else "alive"}
    return rec, data


def train(net, data, epochs: int, lr: float = 1e-3, bs: int = 256, seed: int = 0):
    g = torch.stack([d[0] for d in data])
    s = torch.stack([d[1] for d in data])
    y = torch.tensor([d[2] for d in data])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        perm = torch.randperm(len(y), generator=gen)
        tot, n, correct = 0.0, 0, 0
        for i in range(0, len(y), bs):
            idx = perm[i:i + bs]
            out = net(g[idx], s[idx])
            loss = lossf(out, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
            n += len(idx)
            correct += int((out.argmax(1) == y[idx]).sum())
    return tot / n, correct / n


def summarize(recs: list[dict]) -> str:
    n = len(recs)
    c = collections.Counter(r["cause"] for r in recs)
    return (f"A0 extracted {sum(r['a0_ext'] for r in recs) / n:5.1%}  team {sum(r['team'] for r in recs) / n:5.1%}  "
            f"A0 alive {sum(r['a0_alive'] for r in recs) / n:5.1%}  life {sum(r['life'] for r in recs) / n:5.1f}  "
            f"{dict(c.most_common(4))}")


def eval_seeds(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(0, 999999) for _ in range(n)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", default="camp")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--eps-per-round", type=int, default=200)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--eval-seed", type=int, default=999)
    p.add_argument("--out", default="checkpoints/tiny_camp.pt")
    p.add_argument("--eval", default=None, help="only evaluate this checkpoint")
    p.add_argument("--log", default=None, help="append a JSON line per round here")
    args = p.parse_args(argv)
    torch.manual_seed(0)
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)                      # batch-1 rollouts: one thread is fastest
    seeds_eval = eval_seeds(args.n_eval, args.eval_seed)

    net = TinyNet(args.width)
    n_params = sum(p.numel() for p in net.parameters())
    if args.eval:
        net.load_state_dict(torch.load(args.eval))
        net.eval()
        t = time.time()
        recs = [rollout(s, net_policy(net))[0] for s in seeds_eval]
        dt = time.time() - t
        print(f"tiny ({n_params:,} params) eval N={len(recs)} seed {args.eval_seed}: {summarize(recs)}  ({dt:.0f}s)")
        return
    teacher = get_policy(args.teacher)
    print(f"tiny net: {n_params:,} params; teacher {args.teacher}")
    data: list = []
    rng = random.Random(4242)
    for r in range(args.rounds):
        t = time.time()
        beta = 1.0 if r == 0 else 0.0                 # round 0 = behaviour cloning on teacher rollouts
        net.eval()
        new = []
        for _ in range(args.eps_per_round):
            _, d = rollout(rng.randint(0, 999999), net_policy(net, greedy=False), teacher, beta=beta, rng_key=f"r{r}")
            new.extend(d)
        data.extend(new)
        net.train()
        torch.set_num_threads(n_threads)
        loss, acc = train(net, data, args.epochs, seed=r)
        torch.set_num_threads(1)
        net.eval()
        recs = [rollout(s, net_policy(net))[0] for s in seeds_eval]
        line = f"round {r}: +{len(new)} states (agg {len(data)}), loss {loss:.3f} acc {acc:.1%} | {summarize(recs)}  ({time.time() - t:.0f}s)"
        print(line, flush=True)
        if args.log:
            with open(args.log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"round": r, "teacher": args.teacher, "n_states": len(data), "loss": loss, "acc": acc,
                                    "a0_extracted": sum(x["a0_ext"] for x in recs) / len(recs),
                                    "a0_alive": sum(x["a0_alive"] for x in recs) / len(recs),
                                    "a0_life": sum(x["life"] for x in recs) / len(recs),
                                    "causes": dict(collections.Counter(x["cause"] for x in recs))}) + "\n")
        torch.save(net.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
