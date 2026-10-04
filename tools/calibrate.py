"""Calibrate a BalanceConfig: run scripted policies for all 5 agents, print one table.

    python tools/calibrate.py                          # default preset, 100 eps, all policies
    python tools/calibrate.py --balance v2.2 --tag before
    python tools/calibrate.py --set starve_threshold=20 --set 'wave_schedule={"50":2}'
    python tools/calibrate.py --policies heuristic oracle --n 200 --no-log

Every run (unless --no-log) is documented automatically:
  research_log/experiments.jsonl       one row per policy (metrics, death causes, balance)
  research_log/data/<date>_calibrate_<tag>.json   per-episode records
  research_log/LOG.md                  short entry with the table (skip with --no-md)

Seeds: episode seeds come from random.Random(--seed).randint(0, 999999), the same scheme
as tools/probes/, so --n 100 --seed 42 reproduces the probe numbers.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import logging
import os
import random
import statistics as S
import subprocess
import sys
import time
from typing import Callable

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "probes"))
logging.disable(logging.CRITICAL)

from survivecity_v2_env.balance import BalanceConfig, DEFAULT_PRESET, PRESETS, get_balance  # noqa: E402
from survivecity_v2_env.env import SurviveCityV2Env  # noqa: E402
from training.inference import forage_heuristic_action, random_action  # noqa: E402

LOG_DIR = os.path.join(ROOT, "research_log")


# ---------------------------------------------------------------------------
# Policies: factory(env, seed) -> act(agent_id, obs) -> action dict
# ---------------------------------------------------------------------------

def _rng_policy(fn):
    def factory(env, seed):
        rng = random.Random(seed + 7)
        return lambda aid, obs: fn(aid, obs, rng=rng)
    return factory


def _camp_factory(env, seed):
    from planner2 import Camp
    return Camp(env)


def _oracle_factory(env, seed):
    from oracle import Oracle
    return Oracle(env)


POLICIES: dict[str, Callable] = {
    "random": _rng_policy(random_action),
    "heuristic": _rng_policy(forage_heuristic_action),
    "camp": _camp_factory,
    "oracle": _oracle_factory,
}

POLICY_NOTES = {
    "random": "uniform random actions, random votes",
    "heuristic": "training/inference.py forage_heuristic_action (GRPO rollout policy)",
    "camp": "tools/probes/planner2.py Camp: stock water, camp in safehouse, safe sorties",
    "oracle": "tools/probes/oracle.py: Camp + knows who is infected, votes them out, keeps distance",
}


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------

def episode_seeds(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(0, 999999) for _ in range(n)]


def run_episode(policy: str, balance: BalanceConfig, seed: int, max_actions: int = 5000) -> dict:
    env = SurviveCityV2Env(balance=balance)
    obs = env.reset(seed=seed)
    act = POLICIES[policy](env, seed)
    n = 0
    while not obs.get("done") and n < max_actions:
        aid = obs["metadata"]["current_agent_id"]
        obs = env.step(act(aid, obs))
        n += 1
    return episode_record(env, seed)


def episode_record(env: SurviveCityV2Env, seed: int) -> dict:
    ep = env._episode
    a0 = ep.agents[0]
    healthy = [a for a in ep.agents if a.is_alive and a.infection_state == "none"]
    return {
        "seed": seed,
        "T": ep.step_count,
        "reached_max": ep.step_count >= ep.max_steps,
        "alive_end": sum(a.is_alive for a in ep.agents),
        "healthy_end": len(healthy),
        "survived": len(healthy) >= 1 and ep.step_count >= ep.max_steps,
        "a0_role": a0.infection_role if a0.bite_at_step == 0 else None,
        "a0_life": a0.death_step if a0.death_step is not None else ep.step_count,
        "a0_alive_end": a0.is_alive,
        "a0_death_cause": a0.death_cause,
        "deaths": [
            {"agent": a.agent_id, "cause": a.death_cause, "step": a.death_step,
             "infected": a.infection_state != "none"}
            for a in ep.agents if not a.is_alive
        ],
        "bites": len(ep.bite_history),
        "lockouts": [t for t in ep.lockout_results.values() if t is not None],
        "zombies_end": len(ep.zombies),
    }


def summarize(records: list[dict]) -> dict:
    n = len(records)
    causes = collections.Counter(d["cause"] for r in records for d in r["deaths"])
    healthy_causes = collections.Counter(
        d["cause"] for r in records for d in r["deaths"] if not d["infected"])
    a0_causes = collections.Counter(r["a0_death_cause"] or "alive" for r in records)
    return {
        "metrics": {
            "survival": round(sum(r["survived"] for r in records) / n, 4),
            "reached_max": round(sum(r["reached_max"] for r in records) / n, 4),
            "ep_len": round(S.mean(r["T"] for r in records), 2),
            "a0_life": round(S.mean(r["a0_life"] for r in records), 2),
            "a0_alive_end": round(sum(r["a0_alive_end"] for r in records) / n, 4),
            "healthy_end": round(S.mean(r["healthy_end"] for r in records), 3),
            "alive_end": round(S.mean(r["alive_end"] for r in records), 3),
            "bites": round(S.mean(r["bites"] for r in records), 2),
            "lockouts": round(S.mean(len(r["lockouts"]) for r in records), 2),
        },
        "death_causes": dict(causes.most_common()),
        "healthy_death_causes": dict(healthy_causes.most_common()),
        "a0_outcome": dict(a0_causes.most_common()),
    }


def calibrate(balance: BalanceConfig, policies: list[str], n: int = 100, seed: int = 42,
              progress: bool = True) -> dict[str, dict]:
    seeds = episode_seeds(n, seed)
    out = {}
    for p in policies:
        t0 = time.time()
        recs = [run_episode(p, balance, s) for s in seeds]
        summ = summarize(recs)
        summ["episodes"] = recs
        summ["sec"] = round(time.time() - t0, 1)
        out[p] = summ
        if progress:
            print(f"  {p:<10} done in {summ['sec']}s", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

COLS = [("survival", "surv", "{:.0%}"), ("reached_max", "reach_T", "{:.0%}"),
        ("ep_len", "ep_len", "{:.1f}"), ("a0_life", "A0_life", "{:.1f}"),
        ("healthy_end", "healthy_end", "{:.2f}"), ("alive_end", "alive_end", "{:.2f}"),
        ("bites", "bites", "{:.2f}")]


def format_table(results: dict[str, dict]) -> str:
    head = "| policy | " + " | ".join(c[1] for c in COLS) + " | top death causes (all agents) |"
    sep = "|" + "---|" * (len(COLS) + 2)
    lines = [head, sep]
    for p, r in results.items():
        m = r["metrics"]
        causes = ", ".join(f"{k} {v}" for k, v in list(r["death_causes"].items())[:4]) or "-"
        lines.append(f"| {p} | " + " | ".join(f.format(m[k]) for k, _, f in COLS) + f" | {causes} |")
    return "\n".join(lines)


def git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "--", "survivecity_v2_env", "training", "tools"],
                                cwd=ROOT)
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _unique_path(path: str) -> str:
    base, ext = os.path.splitext(path)
    i = 2
    while os.path.exists(path):
        path = f"{base}_{i}{ext}"
        i += 1
    return path


def write_research_log(results: dict[str, dict], balance: BalanceConfig, balance_name: str,
                       overrides: dict, n: int, seed: int, tag: str, note: str,
                       experiment: str, md: bool) -> str:
    now = dt.datetime.now()
    sha = git_sha()
    os.makedirs(os.path.join(LOG_DIR, "data"), exist_ok=True)
    data_path = _unique_path(os.path.join(
        LOG_DIR, "data", f"{now:%Y-%m-%d}_calibrate_{tag or 'run'}.json"))
    common = {
        "date": now.isoformat(timespec="seconds"), "git_sha": sha, "experiment": experiment,
        "tag": tag, "balance": balance_name, "balance_overrides": overrides,
        "balance_config": balance.to_dict(), "n_eps": n, "seed": seed,
        "max_steps": balance.max_steps, "note": note,
    }
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump({**common, "results": results}, f, indent=1)
    rel_data = os.path.relpath(data_path, LOG_DIR).replace("\\", "/")
    with open(os.path.join(LOG_DIR, "experiments.jsonl"), "a", encoding="utf-8") as f:
        for p, r in results.items():
            row = {**common, "policy": p, "policy_note": POLICY_NOTES.get(p, ""),
                   "metrics": r["metrics"], "death_causes": r["death_causes"],
                   "healthy_death_causes": r["healthy_death_causes"],
                   "a0_outcome": r["a0_outcome"],
                   "episode_lengths": sorted(e["T"] for e in r["episodes"]),
                   "data_file": rel_data}
            f.write(json.dumps(row) + "\n")
    if md:
        ov = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "none"
        entry = (
            f"\n---\n\n## {now:%Y-%m-%d %H:%M} — calibration `{tag or 'run'}` "
            f"(balance `{balance_name}`, overrides: {ov})\n\n"
            + (f"{note}\n\n" if note else "")
            + f"{n} episodes, seed {seed}, git `{sha}`, all 5 agents scripted by the policy. "
              f"surv = >=1 healthy agent alive at t={balance.max_steps}.\n\n"
            + format_table(results)
            + f"\n\nData: `{rel_data}`, `experiments.jsonl` rows with `tag={tag}`.\n"
        )
        with open(os.path.join(LOG_DIR, "LOG.md"), "a", encoding="utf-8") as f:
            f.write(entry)
    return data_path


def parse_overrides(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            val = json.loads(v)
        except json.JSONDecodeError:
            val = v
        if k == "wave_schedule" and isinstance(val, dict):
            val = {int(s): int(c) for s, c in val.items()}
        out[k] = val
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--balance", default=DEFAULT_PRESET, choices=sorted(PRESETS))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                    help="override one BalanceConfig field (repeatable)")
    ap.add_argument("--policies", nargs="+", default=list(POLICIES), choices=list(POLICIES))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="", help="short name for this run (file names, LOG.md)")
    ap.add_argument("--note", default="", help="one line of context for LOG.md / jsonl")
    ap.add_argument("--experiment", default="calibration")
    ap.add_argument("--no-log", action="store_true", help="don't write research_log at all")
    ap.add_argument("--no-md", action="store_true", help="write jsonl + data but no LOG.md entry")
    args = ap.parse_args(argv)

    overrides = parse_overrides(args.set)
    balance = get_balance(args.balance).with_(**overrides) if overrides else get_balance(args.balance)
    name = args.balance + (" + overrides" if overrides else "")
    print(f"balance={name} n={args.n} seed={args.seed}", file=sys.stderr)
    results = calibrate(balance, args.policies, args.n, args.seed)
    print(format_table(results))
    if not args.no_log:
        path = write_research_log(results, balance, args.balance, overrides, args.n, args.seed,
                                  args.tag, args.note, args.experiment, md=not args.no_md)
        print(f"logged -> {os.path.relpath(path, ROOT)}", file=sys.stderr)
    return results


if __name__ == "__main__":
    main()
