"""Record replays of several A0 policies on the same seeds, for side-by-side video footage.

Policies: camp | camp_v2 | tiny:<ckpt.pt> | laya:<ckpt_dir> | laya-search:<ckpt_dir>:<margin>
Teammates camp, A0 healthy, balance v3-rc2. Default seeds = the ones Qwen's final eval recorded
(rc2-k3-r5-final, eval_v3 --seed 4321 --record-replays 4), so all models share footage.

  python tools/record_comparison.py --policy camp --tag cmp-camp
  python tools/record_comparison.py --policy laya-search:/tmp/23ucs715_laya/ckpt/laya_mix2:0.3 --tag cmp-laya-search
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from survivecity_v2_env.balance import get_balance  # noqa: E402
from survivecity_v2_env.env import SurviveCityV2Env  # noqa: E402
from training.policies import get_policy  # noqa: E402
from training.replay import ReplayRecorder  # noqa: E402

QWEN_SEEDS = [266632, 427313, 58523, 837427]


def make_policy(spec: str):
    kind, _, rest = spec.partition(":")
    if kind in ("camp", "camp_v2", "heuristic_v3", "random"):
        return get_policy(kind)
    if kind == "tiny":
        import torch
        from training.tiny_policy import TinyNet, net_policy
        net = TinyNet()
        net.load_state_dict(torch.load(rest))
        net.eval()
        return net_policy(net)
    if kind == "laya":
        from training.laya_policy import LayaPolicy
        return LayaPolicy(rest)
    if kind == "laya-search":
        from training.laya_policy import LayaSearchPolicy
        ckpt, _, margin = rest.rpartition(":")
        return LayaSearchPolicy(ckpt, margin=float(margin))
    raise ValueError(spec)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--seeds", default=",".join(map(str, QWEN_SEEDS)))
    p.add_argument("--out-dir", default="research_log/replays")
    a = p.parse_args()
    pol = make_policy(a.policy)
    mate = get_policy("camp")
    for seed in map(int, a.seeds.split(",")):
        env = SurviveCityV2Env(balance=get_balance("v3-rc2"), a0_healthy=True)
        obs = env.reset(seed=seed)
        rec = ReplayRecorder(env, {"policy": a.policy, "tag": a.tag, "teammates": "camp"})
        rng = random.Random(f"cmp|{seed}")
        while not obs.get("done"):
            aid = obs["metadata"]["current_agent_id"]
            if aid == 0:
                act = (pol(0, obs, rng=rng, env=env) if getattr(pol, "uses_env", False) else pol(0, obs, rng=rng))
            else:
                act = mate(aid, obs, rng=rng)
            obs = env.step(act)
            rec.capture(aid, act)
        ep = env._episode
        a0 = ep.agents[0]
        res = ep.extraction_result or {}
        if getattr(pol, "trace", None) is not None:      # laya-search: per-turn top-2 + lookahead verdicts
            rec.meta["decisions"] = pol.trace
            pol.trace = []
        path = rec.save(os.path.join(a.out_dir, f"{a.tag}_seed{seed}.json"))
        print(json.dumps({"seed": seed, "a0_extracted": 0 in list(res.get("extracted", [])), "a0_alive": a0.is_alive,
                          "life": a0.death_step if a0.death_step is not None else ep.step_count,
                          "cause": a0.death_cause, "file": path}))


if __name__ == "__main__":
    main()
