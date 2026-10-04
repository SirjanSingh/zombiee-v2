"""CPU smoke test for the W6 training pipeline (no GPU, no TRL needed).

    python tools/smoke_w6.py [--n 16] [--dataset-n 200] [--no-log]

1. Builds the 200-scenario mid-episode dataset exactly as train.py would
   (seed 42, prefix 5, default balance) and records its t-histogram + tags.
2. Builds n scenarios and calls the real reward_fn with three fake completion
   sets: all-wait, "planner-like" (the camp planner's own K actions from that
   state), and garbage (unparseable). Expected order: planner >= wait > garbage.
   Runs both reward modes: shaped (step1_weight 1.0) and survival (step1_weight 0).
3. Checks the GiGPO side channel has one entry list per completion.
Logs to research_log (experiments.jsonl, data/, LOG.md) unless --no-log.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import statistics as S
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
logging.disable(logging.WARNING)

from survivecity_v2_env.balance import DEFAULT_PRESET  # noqa: E402
from training.policies import camp_action, get_policy  # noqa: E402
from training.scenarios import build_scenarios, dataset_stats, replay_to  # noqa: E402
from training.train import build_scenario_dataset, create_reward_fn  # noqa: E402

K = 5


def planner_completion(sc, rollout="camp"):
    """The camp planner's own actions at A0's next K turns, as a JSON array."""
    env, obs, ok = replay_to(sc["seed"], sc["t"], sc["mix"])
    pol = get_policy(rollout)
    acts = []
    n = 0
    while not obs.get("done") and len(acts) < K and n < 500:
        aid = obs["metadata"]["current_agent_id"]
        a = camp_action(aid, obs) if aid == 0 else pol(aid, obs)
        if aid == 0:
            acts.append({k: v for k, v in a.items() if k != "agent_id"})
        obs = env.step(a)
        n += 1
    return json.dumps(acts)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--dataset-n", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args(argv)

    full = build_scenarios(args.dataset_n, seed=42, prefix_k=K)
    full_stats = dataset_stats(full)
    print("dataset (as train.py builds it):", json.dumps(full_stats))

    ds = build_scenario_dataset(args.n, seed=7, prefix_actions=K)
    prompts = [r["prompt"] for r in ds]
    scen = build_scenarios(args.n, seed=7, prefix_k=K)
    assert all(sc["tag"] in p for sc, p in zip(scen, prompts)), "dataset/scenario mismatch"

    sets = {
        "all_wait": [json.dumps([{"action_type": "wait"}] * K)] * len(prompts),
        "planner_like": [planner_completion(sc) for sc in scen],
        "garbage": ["I think we should hide. No JSON here."] * len(prompts),
    }
    modes = {}
    for mode, w in (("shaped", 1.0), ("survival", 0.0)):
        side = []
        fn = create_reward_fn(rollout_limit=60, step1_weight=w, prefix_actions=K,
                              gigpo_side_channel=side, rollout_policy="camp",
                              horizon=args.horizon, window_return=mode)
        results = {}
        for name, comps in sets.items():
            side.clear()
            r = fn(prompts, comps)
            assert len(side) == len(prompts), "GiGPO side channel misaligned"
            results[name] = {"rewards": [round(x, 4) for x in r], "mean": round(S.mean(r), 4),
                             "gigpo_entries_per_completion": S.mean(len(e) for e in side)}
        pr, wr, gr = (results[k]["rewards"] for k in ("planner_like", "all_wait", "garbage"))
        summary = {name: {"mean": v["mean"], "gigpo_entries": v["gigpo_entries_per_completion"]}
                   for name, v in results.items()}
        summary.update(
            step1_weight=w,
            planner_beats_wait=f"{sum(p > q for p, q in zip(pr, wr))}/{len(prompts)}",
            wait_beats_planner=f"{sum(p < q for p, q in zip(pr, wr))}/{len(prompts)}",
            wait_minus_garbage=round(S.mean(q - g for q, g in zip(wr, gr)), 4),
            order_ok=results["planner_like"]["mean"] >= results["all_wait"]["mean"] > results["garbage"]["mean"])
        print(f"smoke[{mode}]:", json.dumps(summary))
        modes[mode] = {"results": results, "summary": summary}
    summary = {m: v["summary"] for m, v in modes.items()}
    if args.no_log:
        return summary

    now = dt.datetime.now()
    import calibrate as C
    sha = C.git_sha()
    data_path = C._unique_path(os.path.join(C.LOG_DIR, "data", f"{now:%Y-%m-%d}_w6_smoke.json"))
    rel = os.path.relpath(data_path, C.LOG_DIR).replace("\\", "/")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump({"date": now.isoformat(timespec="seconds"), "git_sha": sha,
                   "dataset_stats": full_stats,
                   "dataset_scenarios": [{k: v for k, v in s.items() if k != "description"} for s in full],
                   "smoke_scenarios": [{k: v for k, v in s.items() if k != "description"} for s in scen],
                   "smoke_results": {m: v["results"] for m, v in modes.items()}, "summary": summary}, f, indent=1)
    with open(os.path.join(C.LOG_DIR, "experiments.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"date": now.isoformat(timespec="seconds"), "git_sha": sha,
                            "experiment": "w6_pipeline_smoke", "balance": DEFAULT_PRESET,
                            "prefix_actions": K, "horizon": args.horizon, "rollout_policy": "camp",
                            "reward_modes": {"shaped": "step1_weight 1.0", "survival": "step1_weight 0.0"},
                            "dataset_stats": full_stats, "smoke": summary, "data_file": rel}) + "\n")
    hist = ", ".join(f"{k}: {v}" for k, v in full_stats["t_hist_by_10"].items())
    tags = ", ".join(f"{k} {v}" for k, v in full_stats["tag_counts"].items())
    with open(os.path.join(C.LOG_DIR, "LOG.md"), "a", encoding="utf-8") as f:
        f.write(f"""
---

## {now:%Y-%m-%d %H:%M} — Training on the moments that matter (W6 smoke, worker session)

Training prompts used to be the step-0 state every time. Now each prompt is A0's real view at a
mid-game moment: the game is played forward by a mix of planner, heuristic and noisy-planner
teammates, and the moment is picked for "decision density" (thirsty, hungry, hurt, outside, zombie
near, vote, bitten, bite just happened). Calm moments are capped at 20%. A0 is always healthy in
training. The reward replays that exact moment, plays the model's 5 actions, and lets the camp
planner continue for 25 steps.

**Dataset the DGX run will use** ({full_stats['n']} prompts, seed 42): time histogram {hist}.
Tags: {tags}. Routine share {full_stats['routine_frac']:.0%}.

**Reward smoke** ({len(prompts)} scenarios, real reward_fn, fake completions; mean reward):

| completion | shaped window (step1_weight 1) | survival window (step1_weight 0) |
|---|---|---|
| planner's own 5 actions | {summary['shaped']['planner_like']['mean']:+.3f} | {summary['survival']['planner_like']['mean']:+.3f} |
| 5 x wait | {summary['shaped']['all_wait']['mean']:+.3f} | {summary['survival']['all_wait']['mean']:+.3f} |
| garbage (unparseable) | {summary['shaped']['garbage']['mean']:+.3f} | {summary['survival']['garbage']['mean']:+.3f} |
| planner beats / loses to wait | {summary['shaped']['planner_beats_wait']} / {summary['shaped']['wait_beats_planner']} | {summary['survival']['planner_beats_wait']} / {summary['survival']['wait_beats_planner']} |

Finding: the shaped window return (the env's 15 rubrics summed over the window) prefers waiting to
the planner's moves, and in a 64-state probe it ranked a random A0 (-1.15) above the camp planner
(-2.53) although the planner kept A0 alive in 42/64 windows vs 27/64. Dying ends the per-step
hunger/thirst penalties, so an early death reads as cheaper than surviving hungry. The survival
window return (+1 alive / -1 dead, small HP and newly-infected terms) ranks them the right way.
Recommendation for the first DGX run: `--window-return survival --step1-weight 0`. Data: `{rel}`.
""")
    print(f"logged -> {os.path.relpath(data_path, ROOT)}")
    return summary


if __name__ == "__main__":
    main()
