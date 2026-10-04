"""Closed-loop eval for v3 training runs (W8).

The trained model drives agent A0 for the WHOLE episode, re-planning every K of
A0's turns (it emits a K-action JSON array, exactly like training). A1-A4 run a
scripted teammate policy (default: camp, matching the training rollouts).
Baselines replace the model with a scripted A0 policy on the SAME seeds and the
SAME teammates, so the comparison is apples to apples.

Prompts are built exactly like training (`build_system_prompt(0, tag + description)`,
fed as raw text with no chat template: TRL 0.15 leaves string prompts untouched).

Episodes run in lockstep: every episode advances until A0 needs a new plan, then
all pending prompts go through ONE batched generate call.

Examples:
  # CPU smoke test, no GPU / model needed
  python -m training.eval_v3 --mock-model --n-episodes 4 --baselines heuristic_v3 --no-log
  # DGX
  CUDA_VISIBLE_DEVICES=5 python -m training.eval_v3 \
      --lora-path checkpoints/run6c_v3/checkpoint-60 --n-episodes 30 --tag run6c-ckpt60
"""

from __future__ import annotations

import argparse
import collections
import copy
import datetime as dt
import json
import logging
import os
import random
import statistics as S
import sys
import time
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from survivecity_v2_env.balance import get_balance  # noqa: E402
from survivecity_v2_env.env import SurviveCityV2Env  # noqa: E402
from survivecity_v2_env.prompts import build_system_prompt  # noqa: E402
from training.inference import parse_actions  # noqa: E402
from training.policies import get_policy  # noqa: E402
from training.scenarios import scenario_tag  # noqa: E402

logger = logging.getLogger("survivecity_v2.eval_v3")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research_log")
WAIT = {"agent_id": 0, "action_type": "wait"}


# ---------------------------------------------------------------------------
# Generators: list of prompt strings -> list of completion strings
# ---------------------------------------------------------------------------

def make_mock_generator(k: int, balance) -> Callable[[list[str], list[dict]], list[str]]:
    """Fake LLM for CPU testing: answers with heuristic_v3's action, repeated K times."""
    pol = get_policy("heuristic_v3")

    def gen(prompts: list[str], obs_list: list[dict]) -> list[str]:
        out = []
        for obs in obs_list:
            a = pol(0, obs, rng=random.Random(0))
            out.append(json.dumps([a] * k))
        return out
    return gen


def make_hf_generator(model_name: str, lora_path: Optional[str], max_new_tokens: int,
                      temperature: float, batch_size: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    if lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device} | base={model_name} | lora={lora_path or 'none (base model)'}")
    model.to(device).eval()

    def gen(prompts: list[str], obs_list: list[dict]) -> list[str]:
        outs: list[str] = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
            if temperature > 0:
                kw.update(do_sample=True, temperature=temperature)
            else:
                kw.update(do_sample=False)
            with torch.no_grad():
                ids = model.generate(**enc, **kw)
            outs += tok.batch_decode(ids[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        return outs
    return gen


# ---------------------------------------------------------------------------
# Lockstep episode runner
# ---------------------------------------------------------------------------

class Episode:
    def __init__(self, seed: int, balance, a0_healthy: bool, teammate: str, k: int,
                 record_meta: Optional[dict] = None, env=None, obs: Optional[dict] = None):
        """env/obs: start from an existing (e.g. replayed mid-episode) state instead of reset(seed)."""
        self.seed = seed
        if env is None:
            self.env = SurviveCityV2Env(balance=balance, a0_healthy=a0_healthy)
            self.obs = self.env.reset(seed=seed)
        else:
            self.env, self.obs = env, obs
        self.balance = balance
        self.mate = get_policy(teammate)
        self.rng = random.Random(f"eval|{seed}")
        self.k = k
        self.queue: list[dict] = []
        self.calls = 0
        self.parse_fail = 0
        self.a0_actions: collections.Counter = collections.Counter()
        self.recorder = None
        if record_meta is not None:
            from training.replay import ReplayRecorder
            self.recorder = ReplayRecorder(self.env, {**record_meta, "teammates": teammate,
                                                      "prefix_actions": k})

    @property
    def done(self) -> bool:
        return bool(self.obs.get("done"))

    def advance(self, a0_policy: Optional[Callable]) -> bool:
        """Step until A0 needs a model plan (returns True) or the episode ends (False).

        a0_policy: scripted A0 (baselines). None = A0 is driven by the model queue.
        """
        while not self.done:
            aid = self.obs["metadata"]["current_agent_id"]
            if aid == 0:
                if a0_policy is not None:
                    act = a0_policy(0, self.obs, rng=self.rng)
                elif self.queue:
                    act = self.queue.pop(0)
                else:
                    return True
                self.a0_actions[act.get("action_type", "?")] += 1
            else:
                act = self.mate(aid, self.obs, rng=self.rng)
            self.obs = self.env.step(act)
            if self.recorder is not None:
                self.recorder.capture(aid, act)
        return False

    def prompt(self) -> str:
        t = self.env._episode.step_count
        desc = self.obs.get("description", "")
        return build_system_prompt(0, f"{scenario_tag(self.seed, t, None)}\n{desc}",
                                   prefix_actions=self.k, balance=self.balance)

    def accept(self, completion: str) -> None:
        self.calls += 1
        acts = parse_actions(completion, agent_id=0, max_actions=self.k)
        if not acts:
            self.parse_fail += 1
            acts = [dict(WAIT)]
        for a in acts:
            a["agent_id"] = 0
        self.queue = acts

    def record(self) -> dict:
        ep = self.env._episode
        a0 = ep.agents[0]
        healthy = [a for a in ep.agents if a.is_alive and a.infection_state == "none"]
        return {
            "seed": self.seed,
            "T": ep.step_count,
            "reached_max": ep.step_count >= ep.max_steps,
            "a0_alive_end": a0.is_alive,
            "a0_survived": a0.is_alive and ep.step_count >= ep.max_steps,
            "a0_life": a0.death_step if a0.death_step is not None else ep.step_count,
            "a0_death_cause": a0.death_cause,
            "healthy_end": len(healthy),
            "survived": len(healthy) >= 1 and ep.step_count >= ep.max_steps,
            "model_calls": self.calls,
            "parse_fail": self.parse_fail,
            "a0_actions": dict(self.a0_actions),
            "deaths": [{"agent": a.agent_id, "cause": a.death_cause, "step": a.death_step}
                       for a in ep.agents if not a.is_alive],
            **extraction_fields(ep),
        }


def extraction_fields(ep) -> dict:
    """W4 extraction outcome (empty when the objective is off)."""
    if not ep.balance.extraction_enabled:
        return {}
    res = ep.extraction_result or {}
    extracted = list(res.get("extracted", []))
    return {
        "extracted": bool(res.get("success")),          # >=1 healthy extracted, flight not failed
        "failed_flight": bool(res.get("failed_flight")),
        "n_extracted": len(extracted),
        "a0_extracted": 0 in extracted,
        "extraction_zone": res.get("zone"),
    }


def run_policy(seeds: list[int], balance, a0_healthy: bool, teammate: str, k: int,
               a0_policy: Optional[Callable] = None, generator: Optional[Callable] = None,
               progress: bool = True, dagger: Optional[dict] = None,
               record_n: int = 0, record_dir: Optional[str] = None, label: str = "") -> list[dict]:
    eps = [Episode(s, balance, a0_healthy, teammate, k,
                   record_meta={"a0_policy": label} if i < record_n else None)
           for i, s in enumerate(seeds)]
    rounds = 0
    while True:
        need = [e for e in eps if not e.done and e.advance(a0_policy)]
        if not need:
            break
        if generator is None:
            raise RuntimeError("A0 needs a model plan but no generator was given")
        prompts = [e.prompt() for e in need]
        if dagger is not None:
            # DAgger: label the states the MODEL reaches with the teacher's plan
            # from that exact state (on a copy, so the live episode is untouched).
            from training.build_sft_dataset import plan_from_env
            for e, p in zip(need, prompts):
                t = e.env._episode.step_count
                plan = plan_from_env(copy.deepcopy(e.env), e.obs, dagger["teacher"], dagger["teammate"],
                                     k, f"dagger|{e.seed}|{t}")
                if len(plan) == k:
                    dagger["rows"].append({"prompt": p, "completion": json.dumps(plan),
                                           "seed": e.seed, "t": t, "source": "dagger"})
        comps = generator(prompts, [e.obs for e in need])
        for e, c in zip(need, comps):
            e.accept(c)
        rounds += 1
        if progress and rounds % 5 == 0:
            alive = sum(not e.done for e in eps)
            logger.info(f"  round {rounds}: {alive}/{len(eps)} episodes running, "
                        f"t~{max(e.env._episode.step_count for e in eps)}")
    if record_n and record_dir:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
        for e in eps:
            if e.recorder is not None:
                path = e.recorder.save(os.path.join(record_dir, f"{safe}_seed{e.seed}.json"))
                logger.info(f"replay saved: {path}")
    return [e.record() for e in eps]


def summarize(records: list[dict]) -> dict:
    n = len(records)
    acts = collections.Counter()
    for r in records:
        acts.update(r["a0_actions"])
    calls = sum(r["model_calls"] for r in records)
    return {
        "metrics": {
            "a0_survived": round(sum(r["a0_survived"] for r in records) / n, 4),
            "a0_life": round(S.mean(r["a0_life"] for r in records), 2),
            "team_survival": round(sum(r["survived"] for r in records) / n, 4),
            "healthy_end": round(S.mean(r["healthy_end"] for r in records), 3),
            "ep_len": round(S.mean(r["T"] for r in records), 2),
            "parse_rate": round(1 - sum(r["parse_fail"] for r in records) / calls, 4) if calls else None,
            "model_calls": calls,
            **({"extraction": round(sum(r["extracted"] for r in records) / n, 4),
                "failed_flight": round(sum(r["failed_flight"] for r in records) / n, 4),
                "n_extracted": round(S.mean(r["n_extracted"] for r in records), 3),
                "a0_extracted": round(sum(r["a0_extracted"] for r in records) / n, 4)}
               if records and "extracted" in records[0] else {}),
        },
        "a0_outcome": dict(collections.Counter(r["a0_death_cause"] or "alive" for r in records)),
        "a0_actions": dict(acts.most_common()),
    }


def format_table(results: dict[str, dict]) -> str:
    ext = any("extraction" in r["metrics"] for r in results.values())
    lines = ["| A0 policy | " + ("extraction | A0 extracted | failed flight | " if ext else "")
             + "A0 survives to end | A0 lifetime | team survival | healthy at end | parse | A0 outcome |",
             "|---|---|---|---|---|---|---|" + ("---|---|---|" if ext else "")]
    for name, r in results.items():
        m = r["metrics"]
        pr = "-" if m["parse_rate"] is None else f"{m['parse_rate']:.0%}"
        out = ", ".join(f"{k} {v}" for k, v in r["a0_outcome"].items())
        ex = (f"{m['extraction']:.0%} | {m['a0_extracted']:.0%} | {m['failed_flight']:.0%} | " if ext else "")
        lines.append(f"| {name} | {ex}{m['a0_survived']:.0%} | {m['a0_life']:.1f} | "
                     f"{m['team_survival']:.0%} | {m['healthy_end']:.2f} | {pr} | {out} |")
    return "\n".join(lines)


def write_log(results: dict, args, seeds: list[int], elapsed: float) -> str:
    import subprocess
    now = dt.datetime.now()
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = "unknown"
    os.makedirs(os.path.join(LOG_DIR, "data"), exist_ok=True)
    tag = args.tag or "eval"
    path = os.path.join(LOG_DIR, "data", f"{now:%Y-%m-%d}_eval_{tag}.json")
    i = 2
    while os.path.exists(path):
        path = os.path.join(LOG_DIR, "data", f"{now:%Y-%m-%d}_eval_{tag}_{i}.json")
        i += 1
    common = {"date": now.isoformat(timespec="seconds"), "git_sha": sha, "experiment": "eval_v3",
              "tag": tag, "balance": args.balance, "teammates": args.teammate_policy,
              "a0_healthy": args.a0_healthy, "prefix_actions": args.prefix_actions,
              "lora_path": args.lora_path, "mock_model": args.mock_model,
              "temperature": args.temperature, "n_eps": len(seeds), "seed": args.seed}
    with open(path, "w", encoding="utf-8") as f:
        json.dump({**common, "seeds": seeds, "results": results}, f, indent=1)
    rel = os.path.relpath(path, LOG_DIR).replace("\\", "/")
    with open(os.path.join(LOG_DIR, "experiments.jsonl"), "a", encoding="utf-8") as f:
        for name, r in results.items():
            f.write(json.dumps({**common, "policy": name, "metrics": r["metrics"],
                                "a0_outcome": r["a0_outcome"], "a0_actions": r["a0_actions"],
                                "data_file": rel}) + "\n")
    entry = (f"\n---\n\n## {now:%Y-%m-%d %H:%M} — eval `{tag}` (closed loop, balance `{args.balance}`)\n\n"
             f"A0 driven by each policy for the whole episode (model re-plans every {args.prefix_actions} "
             f"A0 turns); A1-A4 = `{args.teammate_policy}`; same {len(seeds)} seeds for every row; "
             f"A0 healthy = {args.a0_healthy}; git `{sha}`; {elapsed:.0f} s.\n\n"
             + format_table(results) + f"\n\nData: `{rel}`.\n")
    with open(os.path.join(LOG_DIR, "LOG.md"), "a", encoding="utf-8") as f:
        f.write(entry)
    return path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lora-path", default=None, help="LoRA checkpoint dir; omit for the base model")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--mock-model", action="store_true", help="CPU fake LLM (heuristic_v3 as JSON)")
    p.add_argument("--no-model", action="store_true", help="run baselines only")
    p.add_argument("--baselines", nargs="*", default=["heuristic_v3", "camp", "random", "wait"])
    p.add_argument("--n-episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=1234, help="eval seeds differ from training (42)")
    p.add_argument("--balance", default="v3-rc1")
    p.add_argument("--teammate-policy", default="camp", choices=["camp", "heuristic_v3", "random"])
    p.add_argument("--natural-roles", dest="a0_healthy", action="store_false", default=True)
    p.add_argument("--prefix-actions", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--tag", default="")
    p.add_argument("--dagger-out", default=None,
                   help="write (prompt, teacher plan) JSONL for every state where the model planned")
    p.add_argument("--dagger-teacher", default="camp")
    p.add_argument("--record-replays", type=int, default=0,
                   help="record full replays of the first N episodes of EVERY row (same seeds)")
    p.add_argument("--replay-dir", default=os.path.join(LOG_DIR, "replays"))
    p.add_argument("--no-log", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = parse_args(argv)
    logging.getLogger("survivecity_v2_env").setLevel(logging.WARNING)  # per-step env logs
    balance = get_balance(args.balance)
    rng = random.Random(args.seed)
    seeds = [rng.randint(0, 999999) for _ in range(args.n_episodes)]
    t0 = time.time()
    results: dict[str, dict] = {}

    if not args.no_model:
        if args.mock_model:
            gen, name = make_mock_generator(args.prefix_actions, balance), "mock(heuristic_v3)"
        else:
            gen = make_hf_generator(args.model_name, args.lora_path, args.max_new_tokens,
                                    args.temperature, args.batch_size)
            name = f"model({os.path.basename(os.path.normpath(args.lora_path))})" if args.lora_path else "base model"
        logger.info(f"Running {name} on {len(seeds)} episodes")
        dagger = ({"teacher": args.dagger_teacher, "teammate": args.teammate_policy, "rows": []}
                  if args.dagger_out else None)
        recs = run_policy(seeds, balance, args.a0_healthy, args.teammate_policy,
                          args.prefix_actions, generator=gen, dagger=dagger,
                          record_n=args.record_replays, record_dir=args.replay_dir,
                          label=f"{args.tag or 'eval'}_{name}")
        results[name] = {**summarize(recs), "episodes": recs}
        if dagger is not None:
            os.makedirs(os.path.dirname(args.dagger_out) or ".", exist_ok=True)
            with open(args.dagger_out, "w", encoding="utf-8") as f:
                for row in dagger["rows"]:
                    f.write(json.dumps(row) + "\n")
            logger.info(f"DAgger: wrote {len(dagger['rows'])} teacher-labelled states to {args.dagger_out}")

    for b in args.baselines:
        pol = (lambda aid, obs, rng=None: dict(WAIT)) if b == "wait" else get_policy(b)
        recs = run_policy(seeds, balance, args.a0_healthy, args.teammate_policy,
                          args.prefix_actions, a0_policy=pol, progress=False,
                          record_n=args.record_replays, record_dir=args.replay_dir,
                          label=f"{args.tag or 'eval'}_{b}")
        results[b] = {**summarize(recs), "episodes": recs}

    elapsed = time.time() - t0
    print(format_table(results))
    if not args.no_log:
        path = write_log(results, args, seeds, elapsed)
        logger.info(f"Logged to {path}")
    return results


if __name__ == "__main__":
    main()
