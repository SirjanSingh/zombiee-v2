"""Mid-episode training scenarios (W6) and their deterministic scoring.

A scenario is (seed N, time t, behaviour mix m). Rolling the env forward from
reset(N) with MixPolicy(m, N) to A0's turn at step t reproduces the exact state
(positions, stats, zombies, food timers, env rng), so the prompt can be built
once and the reward function can rebuild the state from the `[SEED:N][T:t][MIX:m]`
tag. Prompts with only `[SEED:N]` (pre-W6) replay to t=0.

Which t: decision density. States where A0 is thirsty/hungry, hurt, outside,
near a zombie, bitten, just saw a bite, or faces a vote are weighted by how
many of those apply; "routine" states (safe in the safehouse, nothing pressing)
are capped at `routine_frac` (~20%) of the dataset.

Scoring (`score_completion`): replay to t, apply the model's K actions at A0's
turns while A1-A4 follow the rollout policy, then everyone follows the rollout
policy until step t + K + H. Window return, then + step1_weight x the
model-step raws, minus the invalid-action penalty:

  window_return="shaped"    A0's raw (rubric) reward accumulated from t to the
                            end of the window (includes the model's own steps).
  window_return="survival"  outcome at the window end: +1 alive / -1 dead,
                            + 0.5 * hp/hp_max if alive, -0.5 if A0 was healthy
                            at t and is infected at the end.

Why "survival" exists: a W6 probe (research_log 2026-10-04) found the shaped
window return ranks A0 behaviours against survival. Over 64 states, random A0
scored -1.15 and the camp planner -2.53, while the planner kept A0 alive in
42/64 windows vs 27/64 for random. Dying ends the per-step hunger/thirst
penalties, so it reads as cheaper than surviving hungry.
"""

from __future__ import annotations

import copy
import random
import re
from collections import Counter, OrderedDict
from typing import Callable, Optional

from survivecity_v2_env.balance import get_balance
from survivecity_v2_env.env import SurviveCityV2Env
from survivecity_v2_env.layout import SAFEHOUSE_CELLS
from training.policies import VOTE_STEPS, MixPolicy, get_policy, sample_mix

TAG_RE = re.compile(r"\[SEED:(\d+)\](?:\[T:(\d+)\])?(?:\[MIX:([a-z]+)\])?")
MAX_REPLAY_ACTIONS = 5000


def scenario_tag(seed: int, t: int, mix: Optional[str]) -> str:
    return f"[SEED:{seed}][T:{t}]" + (f"[MIX:{mix}]" if mix else "")


def parse_scenario_tag(prompt: str) -> Optional[tuple[int, int, Optional[str]]]:
    m = TAG_RE.search(prompt)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2) or 0), m.group(3)


def _a0_turn(obs: dict) -> bool:
    return obs.get("metadata", {}).get("current_agent_id") == 0


def replay_to(seed: int, t: int, mix: Optional[str], balance=None, a0_healthy: bool = True):
    """Rebuild the env at A0's turn of step t. Returns (env, obs, ok).

    ok is False when A0 is dead or the episode ended before t.
    """
    env = SurviveCityV2Env(balance=balance, a0_healthy=a0_healthy)
    obs = env.reset(seed=seed)
    pol = MixPolicy(mix, seed) if mix else None
    n = 0
    while not obs.get("done") and n < MAX_REPLAY_ACTIONS:
        step = env._episode.step_count
        if step > t:
            break
        aid = obs["metadata"]["current_agent_id"]
        if step == t and aid == 0:
            return env, obs, True
        if pol is None:   # legacy [SEED:N] prompt: only t=0 is meaningful
            break
        obs = env.step(pol(aid, obs))
        n += 1
    return env, obs, False


def state_tags(obs: dict, cfg, prefix_k: int = 1) -> list[str]:
    """Why this A0 decision point matters. Empty list = routine."""
    me = next((a for a in obs["agents"] if a["agent_id"] == 0), None)
    if me is None or not me.get("is_alive", True):
        return ["a0_dead"]
    t = obs.get("step_count", 0)
    meta = obs.get("metadata", {}) or {}
    inv = me.get("inventory", []) or []
    pos = (me["row"], me["col"])
    tags = []
    if me["thirst"] >= cfg.dehydrate_threshold // 2 and "water" not in inv:
        tags.append("thirsty")
    if me["hunger"] >= cfg.starve_threshold // 2 and "food" not in inv:
        tags.append("hungry")
    if me["hp"] < cfg.hp_max:
        tags.append("hurt")
    if pos not in SAFEHOUSE_CELLS:
        tags.append("outside")
    if any(abs(z["row"] - pos[0]) + abs(z["col"] - pos[1]) <= 3 for z in obs.get("zombies", [])):
        tags.append("zombie_near")
    if any(t <= v < t + max(1, prefix_k) for v in VOTE_STEPS):
        tags.append("vote")
    if me.get("infection_state") == "latent":
        tags.append("bitten")
    if any(t - b["step"] <= 5 for b in meta.get("bite_history", [])):
        tags.append("bite_recent")
    return tags


def build_scenarios(n: int, seed: int = 42, balance=None, a0_healthy: bool = True,
                    routine_frac: float = 0.2, prefix_k: int = 1, t_max: Optional[int] = None,
                    progress: Optional[Callable] = None) -> list[dict]:
    """Sample n scenarios. Each dict: seed, t, mix, tags, description, tag."""
    cfg = get_balance(balance)
    t_max = cfg.max_steps - 5 if t_max is None else t_max
    rng = random.Random(seed)
    out: list[dict] = []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        ep_seed = rng.randint(0, 999999)
        mix = sample_mix(rng)
        env = SurviveCityV2Env(balance=cfg, a0_healthy=a0_healthy)
        obs = env.reset(seed=ep_seed)
        pol = MixPolicy(mix, ep_seed)
        cands = []
        seen_steps = set()
        k = 0
        while not obs.get("done") and k < MAX_REPLAY_ACTIONS:
            step = env._episode.step_count
            if step > t_max:
                break
            if _a0_turn(obs) and step not in seen_steps:
                seen_steps.add(step)
                cands.append((step, state_tags(obs, cfg, prefix_k), obs["description"]))
            obs = env.step(pol(obs["metadata"]["current_agent_id"], obs))
            k += 1
        routine = [c for c in cands if not c[1]]
        busy = [c for c in cands if c[1]]
        if busy and (not routine or rng.random() >= routine_frac):
            pick = rng.choices(busy, weights=[len(c[1]) for c in busy], k=1)[0]
        elif routine:
            pick = rng.choice(routine)
        else:
            continue
        t, tags, desc = pick
        out.append({"seed": ep_seed, "t": t, "mix": mix, "tags": tags, "description": desc,
                    "tag": scenario_tag(ep_seed, t, mix)})
        if progress:
            progress(len(out))
    return out


def dataset_stats(scenarios: list[dict]) -> dict:
    ts = [s["t"] for s in scenarios]
    bins = Counter((t // 10) * 10 for t in ts)
    tags = Counter(tag for s in scenarios for tag in s["tags"])
    return {
        "n": len(scenarios),
        "t_hist_by_10": {f"{b}-{b + 9}": bins[b] for b in sorted(bins)},
        "tag_counts": dict(tags.most_common()),
        "routine_frac": round(sum(1 for s in scenarios if not s["tags"]) / max(1, len(scenarios)), 3),
        "mix_letter_counts": dict(Counter("".join(s["mix"] for s in scenarios))),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class ReplayCache:
    """Small LRU of replayed envs: a GRPO group shares one (seed, t, mix)."""

    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self._d: OrderedDict = OrderedDict()

    def get(self, key, build):
        if key in self._d:
            self._d.move_to_end(key)
        else:
            self._d[key] = build()
            if len(self._d) > self.maxsize:
                self._d.popitem(last=False)
        env, obs, ok = self._d[key]
        return copy.deepcopy(env), copy.deepcopy(obs), ok


def score_completion(prompt: str, actions: list[dict], *, rollout_policy: str = "camp",
                     horizon: int = 25, balance=None, a0_healthy: bool = True,
                     step1_weight: float = 1.0, parse_ok: bool = True,
                     invalid_action_penalty: float = 0.10, format_bonus: float = 0.0,
                     max_actions: int = 2000, cache: Optional[ReplayCache] = None,
                     anchor_fn: Optional[Callable] = None,
                     window_return: str = "shaped") -> dict:
    """Score K model actions from the scenario in `prompt`. See module docstring."""
    parsed = parse_scenario_tag(prompt)
    seed, t, mix = parsed if parsed else (abs(hash(prompt)) % 1_000_000, 0, None)
    key = (seed, t, mix, a0_healthy, repr(get_balance(balance)))
    build = lambda: replay_to(seed, t, mix, balance=balance, a0_healthy=a0_healthy)  # noqa: E731
    env, obs, ok = cache.get(key, build) if cache is not None else build()
    if not ok:
        raise RuntimeError(f"scenario {scenario_tag(seed, t, mix)} does not replay to an A0 turn")

    policy = get_policy(rollout_policy)
    rng = random.Random(f"rollout|{seed}|{t}")
    cum_start = obs["metadata"]["cumulative_rewards"].get(0, 0.0)
    a0_start_infected = env._episode.agents[0].infection_state != "none"
    t_end = t + len(actions) + horizon
    used = 0
    model_raws: list[float] = []
    entries: list[tuple] = []
    first_breakdown = None
    n = 0
    while not obs.get("done") and env._episode.step_count < t_end and n < max_actions:
        aid = obs["metadata"]["current_agent_id"]
        if aid == 0 and used < len(actions):
            pre_anchor = anchor_fn(obs) if anchor_fn else None
            obs = env.step(actions[used])
            used += 1
            raw = float(obs["metadata"].get("raw_reward", 0.0))
            model_raws.append(raw)
            entries.append((pre_anchor, raw))
            if first_breakdown is None:
                first_breakdown = dict(obs["metadata"].get("rubric_breakdown") or {})
        else:
            obs = env.step(policy(aid, obs, rng=rng))
        n += 1
    if window_return == "survival":
        a0 = env._episode.agents[0]
        if a0.is_alive:
            window = 1.0 + 0.5 * a0.hp / env._episode.balance.hp_max
            if not a0_start_infected and a0.infection_state != "none":
                window -= 0.5
        else:
            window = -1.0
    elif window_return == "graded":
        # Survival, but graded so a GRPO group rarely ties: run 6 attempt 1 showed
        # 5/5 groups with std=0 under the binary alive/dead return (every sample
        # either died in the window or survived at full hp).
        #   dead:  -1 + 0.5 * fraction of the window survived (later death is better)
        #   alive: +1 + 0.5*hp/hp_max + 0.25*food headroom + 0.25*water headroom
        #          (- 0.5 if newly infected); headroom = 1 - meter/threshold, clipped.
        a0 = env._episode.agents[0]
        cfg = env._episode.balance
        span = max(1, t_end - t)
        if a0.is_alive:
            food = 1.0 - min(1.0, a0.hunger / cfg.starve_threshold)
            water = 1.0 - min(1.0, a0.thirst / cfg.dehydrate_threshold)
            window = 1.0 + 0.5 * a0.hp / cfg.hp_max + 0.25 * food + 0.25 * water
            if not a0_start_infected and a0.infection_state != "none":
                window -= 0.5
        else:
            lived = (a0.death_step if a0.death_step is not None else t) - t
            window = -1.0 + 0.5 * max(0.0, min(1.0, lived / span))
    elif window_return == "shaped":
        window = obs["metadata"]["cumulative_rewards"].get(0, 0.0) - cum_start
    else:
        raise ValueError(f"unknown window_return {window_return!r}")
    if window_return in ("survival", "graded"):
        # GiGPO step reward = return from that step on. Every model step in the
        # prefix shares the window outcome; the shaped per-step raws would bring
        # the misaligned rubric signal back in through the step advantage.
        entries = [(anchor, float(window)) for anchor, _ in entries]
    composite = (step1_weight * sum(model_raws) + window
                 + (format_bonus if parse_ok else -invalid_action_penalty))
    return {
        "reward": float(composite), "window_return": float(window), "model_raws": model_raws,
        "gigpo_entries": entries, "final_obs": obs, "steps": n, "rubric_breakdown": first_breakdown,
        "seed": seed, "t": t, "mix": mix, "t_end": env._episode.step_count,
    }
