"""Build the SFT warm-start dataset (W7): imitate the camp planner.

For each mid-episode scenario (same distribution as the RL dataset, see
training/scenarios.py) the prompt is EXACTLY the RL prompt and the completion is
the K actions the camp planner actually takes on A0's next K turns (teammates
also run camp, as in the RL rollouts), written in the prompt's JSON-array format.

Why: the untrained base model plays worse than waiting every turn (closed-loop
A0 lifetime 20.8 vs 24.0; camp planner 94.5). RL from that start is slow, so we
first teach the planner's behaviour, then let RL improve on it.

  python -m training.build_sft_dataset --n 2000 --out data/sft_camp_v3.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from survivecity_v2_env.balance import get_balance  # noqa: E402
from survivecity_v2_env.prompts import build_system_prompt  # noqa: E402
from training.policies import get_policy  # noqa: E402
from training.scenarios import build_scenarios, replay_to  # noqa: E402


def compact(action: dict) -> dict:
    """Drop agent_id and empty fields: {"action_type": "pickup", "item_type": "water"}."""
    return {k: v for k, v in action.items() if k != "agent_id" and v is not None}


def plan_from_env(env, obs: dict, teacher: str, teammate: str, k: int, rng_key: str) -> list[dict]:
    """The teacher's actions on A0's next k turns from the env's CURRENT state.

    `env` is advanced in place, so pass a copy (copy.deepcopy) when the caller
    still needs the original. Teammates run `teammate`, as in the RL rollouts.
    """
    pol, mate = get_policy(teacher), get_policy(teammate)
    rng = random.Random(rng_key)
    plan: list[dict] = []
    guard = 0
    while len(plan) < k and not obs.get("done") and guard < 200:
        guard += 1
        aid = obs["metadata"]["current_agent_id"]
        if aid == 0:
            act = pol(0, obs, rng=rng)
            plan.append(compact(act))
        else:
            act = mate(aid, obs, rng=rng)
        obs = env.step(act)
    return plan


def teacher_plan(sc: dict, balance, teacher: str, teammate: str, k: int,
                 a0_healthy: bool = True) -> list[dict]:
    """The teacher's actions on A0's next k turns, starting from the scenario state."""
    env, obs, ok = replay_to(sc["seed"], sc["t"], sc["mix"], balance=balance, a0_healthy=a0_healthy)
    if not ok:
        return []
    return plan_from_env(env, obs, teacher, teammate, k, f"sft|{sc['seed']}|{sc['t']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=777, help="differs from RL (42) and eval (1234)")
    p.add_argument("--balance", default="v3-rc1")
    p.add_argument("--prefix-actions", type=int, default=5)
    p.add_argument("--routine-frac", type=float, default=0.2)
    p.add_argument("--teacher", default="camp")
    p.add_argument("--teammate-policy", default="camp")
    p.add_argument("--out", default="data/sft_camp_v3.jsonl")
    args = p.parse_args(argv)

    balance = get_balance(args.balance)
    scen = build_scenarios(args.n, seed=args.seed, balance=balance, a0_healthy=True,
                           routine_frac=args.routine_frac, prefix_k=args.prefix_actions)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    acts = collections.Counter()
    kept = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for sc in scen:
            plan = teacher_plan(sc, balance, args.teacher, args.teammate_policy, args.prefix_actions)
            if len(plan) < args.prefix_actions:
                continue  # episode ended inside the plan; skip rather than pad
            prompt = build_system_prompt(0, f"{sc['tag']}\n{sc['description']}",
                                         prefix_actions=args.prefix_actions, balance=balance)
            f.write(json.dumps({"prompt": prompt, "completion": json.dumps(plan),
                                "seed": sc["seed"], "t": sc["t"], "tags": sc["tags"]}) + "\n")
            acts.update(a["action_type"] for a in plan)
            kept += 1
    total = sum(acts.values())
    print(f"wrote {kept}/{len(scen)} examples to {args.out}")
    print("teacher action mix:", ", ".join(f"{a} {c / total:.0%}" for a, c in acts.most_common()))


if __name__ == "__main__":
    main()
