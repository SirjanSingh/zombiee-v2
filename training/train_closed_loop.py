"""Closed-loop multi-turn RL trainer (run 8): the model plays A0 at EVERY decision.

Why: training/train.py (TRL GRPOTrainer) is one-shot. The model writes one K-action
plan from a mid-game state and a scripted policy plays the rest, so it is a
contextual bandit with a heuristic continuation. Runs 6c/7 barely moved the
policy. Here every A0 decision in a rollout is a model sample and gets its own
advantage (GiGPO / GAGPO style turn-level credit).

One training step:
  1. Sample B start states from the training/scenarios.py distribution
     (decision-dense mid-episode states, A0 healthy) and replay each one.
  2. Run G rollouts per start state (deepcopy of the replayed env). The model
     drives A0 and re-plans every K of A0's turns; A1-A4 run the teammate policy
     (camp). A rollout ends after H game turns, when A0 dies, or at game end.
     All B*G rollouts run in lockstep (training/eval_v3.Episode machinery): every
     rollout advances until A0 needs a plan, then ONE batched generate call, in
     eval mode (KV cache on; see gigpo_trainer.run_in_eval_mode).
  3. Per-decision rewards (--reward):
       graded  0 at intermediate decisions; at the last decision the graded window
               value of training/scenarios.score_completion: alive -> 1 + 0.5 hp
               + 0.25 food + 0.25 water headroom (-0.5 if newly infected),
               dead -> -1 + 0.5 * fraction of the horizon survived.
       env     A0's cumulative env reward gained over the decision's segment
               (raw rubric reward + terminal terms).
  4. Advantages (--adv-estimator), per decision t of rollout i in start group g:
       grpo    A = (R_i - mean_g R) / (std_g R + eps), R_i = sum of the rollout's
               rewards; the same value for every decision of the rollout.
       gigpo   grpo + w * step advantage: discounted return-to-go G_t, normalised
               inside (start group, anchor_key_for_agent0) clusters
               (training/gigpo.step_norm_reward; mean-subtract, or z-score with
               --gigpo-zscore). Singleton clusters get 0.
       gagpo   V(s) = mean G_t over every decision in the batch whose anchor key
               equals s (exact key). delta_t = r_t + gamma V(s_{t+1}) - V(s_t)
               (V = 0 after the last decision), A_t = delta_t + gamma lambda A_{t+1}
               (GAE), then standardised per start group.
  5. Loss on completion tokens: PPO-clip (or REINFORCE) + beta * KL(k3) to the
     reference policy = the frozen init adapter (keeps RL near the SFT/DAgger
     policy); with a fresh LoRA the reference is the base model. fp16 autocast,
     fp32 LoRA params, GradScaler, AdamW (not fused: V100 is sm_70), gradient
     checkpointing during the update only.

Prompts: exactly eval_v3's (`build_system_prompt(0, "[SEED:N][T:t]\\n" + description)`,
raw text, no chat template), the distribution the DAgger adapter was trained on.

Examples:
  # DGX (see README "Run 8")
  CUDA_VISIBLE_DEVICES=5 python -m training.train_closed_loop \\
      --init-adapter checkpoints/sft_dagger2 --adv-estimator gagpo \\
      --output-dir checkpoints/run8_gagpo
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import inspect
import json
import logging
import math
import os
import random
import re
import shutil
import statistics as S
import sys
import time
from collections import defaultdict
from typing import Callable, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from survivecity_v2_env.balance import get_balance  # noqa: E402
from training.eval_v3 import Episode  # noqa: E402
from training.gigpo import (anchor_key_for_agent0, build_step_group,  # noqa: E402
                            cluster_size_summary, step_norm_reward)
from training.gigpo_trainer import run_in_eval_mode  # noqa: E402
from training.scenarios import build_scenarios, dataset_stats, replay_to  # noqa: E402

logger = logging.getLogger("survivecity_v2.closed_loop")
GROUP_EPS = 1e-4          # same epsilon as TRL GRPO's group normalisation
ZERO_VAR = 1e-6


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

def graded_terminal(env, t0: int, span: int, a0_start_infected: bool) -> float:
    """Graded window value. Same formula as scenarios.score_completion(window_return="graded")."""
    a0 = env._episode.agents[0]
    cfg = env._episode.balance
    if a0.is_alive:
        food = 1.0 - min(1.0, a0.hunger / cfg.starve_threshold)
        water = 1.0 - min(1.0, a0.thirst / cfg.dehydrate_threshold)
        v = 1.0 + 0.5 * a0.hp / cfg.hp_max + 0.25 * food + 0.25 * water
        if not a0_start_infected and a0.infection_state != "none":
            v -= 0.5
        return v
    lived = (a0.death_step if a0.death_step is not None else t0) - t0
    return -1.0 + 0.5 * max(0.0, min(1.0, lived / max(1, span)))


# ---------------------------------------------------------------------------
# Rollouts (lockstep, built on eval_v3.Episode)
# ---------------------------------------------------------------------------

class Rollout(Episode):
    """One closed-loop rollout from a replayed start state, with per-decision records."""

    def __init__(self, env, obs, *, seed: int, group: int, balance, teammate: str, k: int,
                 horizon: int, reward_mode: str):
        super().__init__(seed, balance, True, teammate, k, env=env, obs=obs)
        self.group = group
        self.t0 = env._episode.step_count
        self.t_end = self.t0 + horizon
        self.horizon = horizon
        self.reward_mode = reward_mode
        # Same teammate rng for every copy of a start state: rollouts differ only
        # through the model's actions.
        self.rng = random.Random(f"train|{seed}|{self.t0}")
        self.a0_start_infected = env._episode.agents[0].infection_state != "none"
        self.decisions: list[dict] = []
        self._cum_start = self._cum_at = self._cum()
        self.finished = False

    @property
    def a0(self):
        return self.env._episode.agents[0]

    @property
    def done(self) -> bool:
        return (bool(self.obs.get("done")) or not self.a0.is_alive
                or self.env._episode.step_count >= self.t_end)

    def _cum(self) -> float:
        return float(self.obs.get("metadata", {}).get("cumulative_rewards", {}).get(0, 0.0))

    def _close_decision(self) -> None:
        if self.decisions and self.reward_mode == "env":
            self.decisions[-1]["reward"] = self._cum() - self._cum_at

    def begin_decision(self, comp: dict) -> None:
        """A0 needs a plan: close the previous decision, record and apply this one."""
        self._close_decision()
        before = self.parse_fail
        anchor = anchor_key_for_agent0(self.obs)
        self.accept(comp["text"])
        self.decisions.append({
            "anchor": anchor, "t": self.env._episode.step_count, "reward": 0.0,
            "parsed": self.parse_fail == before,
            "prompt_ids": comp["prompt_ids"], "completion_ids": comp["completion_ids"],
        })
        self._cum_at = self._cum()

    def finish(self) -> None:
        if self.finished:
            return
        self.finished = True
        self._close_decision()
        if self.reward_mode == "graded" and self.decisions:
            self.decisions[-1]["reward"] += graded_terminal(self.env, self.t0, self.horizon,
                                                            self.a0_start_infected)

    def a0_life(self) -> int:
        """Turns A0 lived inside this rollout's window."""
        if not self.a0.is_alive and self.a0.death_step is not None:
            return self.a0.death_step - self.t0
        return self.env._episode.step_count - self.t0

    def as_traj(self) -> dict:
        return {"group": self.group, "rewards": [d["reward"] for d in self.decisions],
                "anchors": [d["anchor"] for d in self.decisions]}


def collect_rollouts(rollouts: list[Rollout], generator: Callable) -> dict:
    """Run all rollouts in lockstep. generator(prompts, obs_list) -> list of
    {"text", "prompt_ids", "completion_ids"}."""
    rounds = gen_time = 0.0
    n_tok = 0
    while True:
        need = [r for r in rollouts if not r.done and r.advance(None)]
        if not need:
            break
        t = time.time()
        comps = generator([r.prompt() for r in need], [r.obs for r in need])
        gen_time += time.time() - t
        for r, c in zip(need, comps):
            n_tok += len(c["completion_ids"])
            r.begin_decision(c)
        rounds += 1
    for r in rollouts:
        r.finish()
    return {"rounds": int(rounds), "gen_time": gen_time, "gen_tokens": n_tok}


# ---------------------------------------------------------------------------
# Advantages (pure functions on toy-friendly trajectories)
# ---------------------------------------------------------------------------

def discounted_returns(rewards: Sequence[float], gamma: float) -> list[float]:
    out, g = [0.0] * len(rewards), 0.0
    for t in range(len(rewards) - 1, -1, -1):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


def _group_standardize(values: list[float], groups: list[int]) -> list[float]:
    """(x - mean_g) / (std_g + eps) with the sample std (TRL GRPO convention).
    Groups with fewer than 2 members or zero variance get 0."""
    by = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    out = [0.0] * len(values)
    for idx in by.values():
        xs = [values[i] for i in idx]
        if len(xs) < 2:
            continue
        sd = S.stdev(xs)
        if sd < ZERO_VAR:
            continue
        m = S.fmean(xs)
        for i in idx:
            out[i] = (values[i] - m) / (sd + GROUP_EPS)
    return out


def grpo_advantages(trajs: list[dict]) -> list[list[float]]:
    returns = [sum(tr["rewards"]) for tr in trajs]
    a = _group_standardize(returns, [tr["group"] for tr in trajs])
    return [[a[i]] * len(tr["rewards"]) for i, tr in enumerate(trajs)]


def gigpo_advantages(trajs: list[dict], gamma: float, step_w: float = 1.0,
                     zscore: bool = False) -> tuple[list[list[float]], dict]:
    ep = grpo_advantages(trajs)
    flat_g, flat_a, flat_grp, owner = [], [], [], []
    for i, tr in enumerate(trajs):
        for t, g in enumerate(discounted_returns(tr["rewards"], gamma)):
            flat_g.append(g)
            flat_a.append(tr["anchors"][t])
            flat_grp.append(tr["group"])
            owner.append((i, t))
    if not flat_g:
        return ep, cluster_size_summary([])
    cids = build_step_group(flat_a, flat_grp)
    step = step_norm_reward(torch.tensor(flat_g, dtype=torch.float32), cids,
                            remove_std=not zscore).tolist()
    out = [list(x) for x in ep]
    for (i, t), s in zip(owner, step):
        out[i][t] += step_w * s
    diag = cluster_size_summary(cids)
    diag["step_adv_std"] = float(torch.tensor(step).std(unbiased=False)) if len(step) > 1 else 0.0
    return out, diag


def gagpo_values(trajs: list[dict], gamma: float) -> dict[str, float]:
    """V(s) = mean discounted return-to-go over every decision in the batch at anchor s."""
    acc = defaultdict(list)
    for tr in trajs:
        for a, g in zip(tr["anchors"], discounted_returns(tr["rewards"], gamma)):
            acc[a].append(g)
    return {a: S.fmean(v) for a, v in acc.items()}


def gagpo_advantages(trajs: list[dict], gamma: float, lam: float,
                     standardize: bool = True) -> tuple[list[list[float]], dict]:
    V = gagpo_values(trajs, gamma)
    raw = []
    for tr in trajs:
        r, an = tr["rewards"], tr["anchors"]
        T = len(r)
        adv = [0.0] * T
        nxt = 0.0
        for t in range(T - 1, -1, -1):
            v_next = V[an[t + 1]] if t + 1 < T else 0.0   # rollout end = terminal
            delta = r[t] + gamma * v_next - V[an[t]]
            nxt = delta + gamma * lam * nxt
            adv[t] = nxt
        raw.append(adv)
    shared = sum(1 for tr in trajs for _ in tr["anchors"])
    diag = {"n_states": len(V), "mean_state_visits": round(shared / max(1, len(V)), 2)}
    if not standardize:
        return raw, diag
    flat = [a for adv in raw for a in adv]
    grp = [tr["group"] for tr, adv in zip(trajs, raw) for _ in adv]
    z = _group_standardize(flat, grp)
    out, k = [], 0
    for adv in raw:
        out.append(z[k:k + len(adv)])
        k += len(adv)
    return out, diag


def compute_advantages(trajs: list[dict], estimator: str, gamma: float = 0.95, lam: float = 0.8,
                       step_w: float = 1.0, gigpo_zscore: bool = False) -> tuple[list[list[float]], dict]:
    if estimator == "grpo":
        return grpo_advantages(trajs), {}
    if estimator == "gigpo":
        return gigpo_advantages(trajs, gamma, step_w, gigpo_zscore)
    if estimator == "gagpo":
        return gagpo_advantages(trajs, gamma, lam)
    raise ValueError(f"unknown estimator {estimator!r}")


def group_return_stats(trajs: list[dict]) -> dict:
    by = defaultdict(list)
    for tr in trajs:
        by[tr["group"]].append(sum(tr["rewards"]))
    stds = [S.pstdev(v) if len(v) > 1 else 0.0 for v in by.values()]
    allr = [x for v in by.values() for x in v]
    return {
        "return_mean": S.fmean(allr) if allr else 0.0,
        "return_std": S.pstdev(allr) if len(allr) > 1 else 0.0,
        "group_return_std_mean": S.fmean(stds) if stds else 0.0,
        "zero_var_group_frac": sum(s < ZERO_VAR for s in stds) / max(1, len(stds)),
    }


# ---------------------------------------------------------------------------
# Model, generation, log-probs
# ---------------------------------------------------------------------------

def eos_ids_for(model, tok) -> set[int]:
    ids = set()
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(gen_eos, int):
        ids.add(gen_eos)
    elif gen_eos:
        ids.update(gen_eos)
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return ids


def make_policy_generator(model, tok, *, max_new_tokens: int, temperature: float, batch_size: int,
                          max_prompt_tokens: int = 0, amp: bool = False) -> Callable:
    """Batched sampling from the current policy, in eval mode (KV cache on).

    top_k/top_p/repetition_penalty are pinned to neutral values: Qwen's
    generation_config ships top_p=0.8, top_k=20, repetition_penalty=1.05, which
    would make the samples come from a different distribution than the policy
    whose log-probs the loss uses.
    """
    eos = eos_ids_for(model, tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else next(iter(eos))

    def _run(enc: list[list[int]]) -> list[dict]:
        device = next(model.parameters()).device
        outs: list[dict] = []
        for i in range(0, len(enc), batch_size):
            chunk = enc[i:i + batch_size]
            L = max(len(x) for x in chunk)
            ids = torch.full((len(chunk), L), pad, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for j, x in enumerate(chunk):
                ids[j, L - len(x):] = torch.tensor(x, dtype=torch.long)
                att[j, L - len(x):] = 1
            kw = dict(max_new_tokens=max_new_tokens, pad_token_id=pad, eos_token_id=sorted(eos),
                      use_cache=True, repetition_penalty=1.0)
            if temperature > 0:
                kw.update(do_sample=True, temperature=temperature, top_k=0, top_p=1.0)
            else:
                kw.update(do_sample=False, temperature=None, top_k=None, top_p=None)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out = model.generate(input_ids=ids.to(device), attention_mask=att.to(device), **kw)
            for j, x in enumerate(chunk):
                comp = out[j, L:].tolist()
                for n, tid in enumerate(comp):
                    if tid in eos:
                        comp = comp[:n + 1]
                        break
                outs.append({"text": tok.decode(comp, skip_special_tokens=True),
                             "prompt_ids": list(x), "completion_ids": comp})
        return outs

    def gen(prompts: list[str], obs_list=None) -> list[dict]:
        enc = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        if max_prompt_tokens:
            enc = [x[-max_prompt_tokens:] for x in enc]
        return run_in_eval_mode(model, _run, enc)
    return gen


def make_scripted_generator(tok, k: int, policy: str = "heuristic_v3",
                            max_prompt_tokens: int = 0) -> Callable:
    """CPU test path: a scripted policy's action as the completion text, tokenized
    (+EOS) so the loss still runs on real token ids."""
    from training.policies import get_policy
    pol = get_policy(policy)

    def gen(prompts: list[str], obs_list: list[dict]) -> list[dict]:
        out = []
        for p, obs in zip(prompts, obs_list):
            text = json.dumps([pol(0, obs, rng=random.Random(0))] * k)
            ids = tok(p, add_special_tokens=False)["input_ids"]
            if max_prompt_tokens:
                ids = ids[-max_prompt_tokens:]
            out.append({"text": text, "prompt_ids": ids,
                        "completion_ids": tok(text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]})
        return out
    return gen


def logits_kwarg(model) -> Optional[str]:
    """transformers renamed num_logits_to_keep -> logits_to_keep; support both."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    params = inspect.signature(base.forward).parameters
    for key in ("logits_to_keep", "num_logits_to_keep"):
        if key in params:
            return key
    return None


def completion_logps(model, samples: list[dict], *, temperature: float = 1.0, amp: bool = False,
                     logits_key: Optional[str] = None, pad_id: int = 0):
    """Per-token log-probs of each sample's completion. Prompts left-padded,
    completions right-padded, position ids from the attention mask (matches
    generate()). Returns (logp [n, C] float32, mask [n, C] float32)."""
    device = next(model.parameters()).device
    P = max(len(s["prompt_ids"]) for s in samples)
    C = max(len(s["completion_ids"]) for s in samples)
    n = len(samples)
    ids = torch.full((n, P + C), pad_id, dtype=torch.long)
    att = torch.zeros((n, P + C), dtype=torch.long)
    cmask = torch.zeros((n, C), dtype=torch.float32)
    for j, s in enumerate(samples):
        p, c = s["prompt_ids"], s["completion_ids"]
        ids[j, P - len(p):P] = torch.tensor(p, dtype=torch.long)
        ids[j, P:P + len(c)] = torch.tensor(c, dtype=torch.long)
        att[j, P - len(p):P + len(c)] = 1
        cmask[j, :len(c)] = 1.0
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    ids, att, pos, cmask = ids.to(device), att.to(device), pos.to(device), cmask.to(device)
    kw = {logits_key: C + 1} if logits_key else {}
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        out = model(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=False, **kw)
    logits = out.logits[:, -(C + 1):-1, :]
    tgt = ids[:, P:]
    rows = []
    for j in range(n):   # row loop: avoids an [n, C, V] fp32 copy
        lj = logits[j].float()
        if temperature not in (0.0, 1.0):
            lj = lj / temperature
        rows.append(lj.gather(-1, tgt[j].unsqueeze(-1)).squeeze(-1) - torch.logsumexp(lj, dim=-1))
    return torch.stack(rows), cmask


@contextlib.contextmanager
def reference_policy(model, ref_mode: str):
    """ref_mode 'adapter': the frozen 'ref' adapter; 'base': LoRA disabled."""
    if ref_mode == "base":
        with model.disable_adapter():
            yield
        return
    model.set_adapter("ref")
    try:
        yield
    finally:
        model.set_adapter("default")   # also restores requires_grad on 'default' only


def lora_config(r: int, alpha: int):
    from peft import LoraConfig
    return LoraConfig(r=r, lora_alpha=alpha, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")


def prepare_policy(base_model, *, init_adapter: Optional[str], ref_adapter: Optional[str],
                   lora_r: int = 64, lora_alpha: int = 128, grad_ckpt: bool = True):
    """Wrap a base model with the trainable LoRA ('default') and the reference.

    Returns (peft_model, ref_mode). ref_mode is 'adapter' when a reference adapter
    is loaded as the frozen 'ref' adapter (default: the init adapter), else 'base'
    (fresh LoRA: B=0 at init, so base == the init policy).
    """
    from peft import PeftModel, get_peft_model
    if init_adapter:
        model = PeftModel.from_pretrained(base_model, init_adapter, is_trainable=True)
    else:
        model = get_peft_model(base_model, lora_config(lora_r, lora_alpha))
    ref = ref_adapter if ref_adapter is not None else init_adapter
    ref_mode = "base"
    if ref:
        model.load_adapter(ref, adapter_name="ref", is_trainable=False)
        model.set_adapter("default")
        ref_mode = "adapter"
    for name, p in model.named_parameters():
        if ".ref." in name:
            p.requires_grad_(False)
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()            # fp32 master LoRA weights for GradScaler
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model, ref_mode


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------

def policy_update(model, optimizer, scaler, samples: list[dict], *, ref_mode: Optional[str], beta: float,
                  loss_type: str = "ppo", clip_eps: float = 0.2, ppo_epochs: int = 1,
                  micro_batch: int = 4, max_grad_norm: float = 1.0, temperature: float = 1.0,
                  amp: bool = False, pad_id: int = 0) -> dict:
    """PPO-clip / REINFORCE on completion tokens + beta * KL(k3) to the reference.
    Loss per sample = mean over its completion tokens; batch loss = mean over samples.
    One optimizer step per epoch (gradients accumulated over micro-batches)."""
    if not samples:
        return {"loss": 0.0, "pg_loss": 0.0, "kl": 0.0, "clip_frac": 0.0, "grad_norm": 0.0}
    lk = logits_kwarg(model)
    order = sorted(range(len(samples)), key=lambda i: len(samples[i]["prompt_ids"]) + len(samples[i]["completion_ids"]))
    mbs = [[samples[i] for i in order[k:k + micro_batch]] for k in range(0, len(order), micro_batch)]
    N = len(samples)
    need_ref = beta > 0 and ref_mode is not None
    need_old = ppo_epochs > 1

    def _no_grad_logps():
        refs, olds = [], []
        with torch.no_grad():
            for mb in mbs:
                if need_ref:
                    with reference_policy(model, ref_mode):
                        refs.append(completion_logps(model, mb, temperature=temperature, amp=amp,
                                                     logits_key=lk, pad_id=pad_id)[0])
                if need_old:
                    olds.append(completion_logps(model, mb, temperature=temperature, amp=amp,
                                                 logits_key=lk, pad_id=pad_id)[0])
        return refs, olds
    refs, olds = run_in_eval_mode(model, _no_grad_logps) if (need_ref or need_old) else ([], [])

    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    stats = defaultdict(float)
    for epoch in range(ppo_epochs):
        optimizer.zero_grad(set_to_none=True)
        for b, mb in enumerate(mbs):
            logp, mask = completion_logps(model, mb, temperature=temperature, amp=amp,
                                          logits_key=lk, pad_id=pad_id)
            adv = torch.tensor([s["advantage"] for s in mb], dtype=torch.float32, device=logp.device)[:, None]
            old = olds[b] if need_old else logp.detach()
            ratio = torch.exp(logp - old)
            if loss_type == "ppo":
                pg = -torch.min(ratio * adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv)
            else:
                pg = -adv * logp
            per_tok = pg
            if need_ref:
                d = refs[b] - logp
                kl = torch.exp(d) - d - 1
                per_tok = per_tok + beta * kl
            denom = mask.sum(1).clamp(min=1)
            per_sample = (per_tok * mask).sum(1) / denom
            loss = per_sample.sum() / N
            scaler.scale(loss).backward()
            if epoch == 0:
                stats["loss"] += loss.item()
                stats["pg_loss"] += (((pg * mask).sum(1) / denom).sum() / N).item()
                if need_ref:
                    stats["kl"] += (((kl.detach() * mask).sum(1) / denom).sum() / N).item()
                clipped = ((ratio.detach() - 1).abs() > clip_eps).float()
                stats["clip_frac"] += (((clipped * mask).sum(1) / denom).sum() / N).item()
        scaler.unscale_(optimizer)
        gn = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        if epoch == 0:
            stats["grad_norm"] = float(gn)
    return dict(stats)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def save_adapter(model, out_dir: str, step: int, keep: int) -> str:
    path = os.path.join(out_dir, f"checkpoint-{step}")
    model.save_pretrained(path, selected_adapters=["default"])
    ckpts = sorted((int(m.group(1)), d) for d in os.listdir(out_dir)
                   if (m := re.fullmatch(r"checkpoint-(\d+)", d)))
    for _, d in ckpts[:max(0, len(ckpts) - keep)]:
        shutil.rmtree(os.path.join(out_dir, d), ignore_errors=True)
    return path


def train(args, model, tok, ref_mode: Optional[str], generator: Optional[Callable] = None) -> list[dict]:
    """Run args.max_steps closed-loop RL steps. Returns the logged rows."""
    os.makedirs(args.output_dir, exist_ok=True)
    cuda = next(model.parameters()).is_cuda
    amp = cuda and not args.no_amp
    balance = get_balance(args.balance)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if generator is None:
        generator = make_policy_generator(model, tok, max_new_tokens=args.max_new_tokens,
                                          temperature=args.temperature, batch_size=args.gen_batch_size,
                                          max_prompt_tokens=args.max_prompt_tokens, amp=amp)

    t = time.time()
    pool = build_scenarios(args.num_scenarios, seed=args.scenario_seed, balance=balance,
                           a0_healthy=True, routine_frac=args.routine_frac, prefix_k=args.prefix_actions)
    stats = dataset_stats(pool)
    logger.info(f"[scenarios] {len(pool)} start states in {time.time() - t:.1f}s: {stats}")
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "ref_mode": ref_mode, "scenario_stats": stats}, f, indent=1)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, fused=False)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    rows: list[dict] = []
    order: list[dict] = []
    for step in range(1, args.max_steps + 1):
        t_step = time.time()
        # 1. start states (cycle through a shuffled pool)
        starts = []
        while len(starts) < args.batch_starts:
            if not order:
                order = pool[:]
                rng.shuffle(order)
            sc = order.pop()
            env, obs, ok = replay_to(sc["seed"], sc["t"], sc["mix"], balance=balance, a0_healthy=True)
            if ok:
                starts.append((sc, env, obs))
        rollouts = [Rollout(copy.deepcopy(env), copy.deepcopy(obs), seed=sc["seed"], group=g,
                            balance=balance, teammate=args.teammate_policy, k=args.prefix_actions,
                            horizon=args.horizon, reward_mode=args.reward)
                    for g, (sc, env, obs) in enumerate(starts) for _ in range(args.group_size)]
        # 2. closed-loop rollouts (generation in eval mode inside the generator)
        roll = collect_rollouts(rollouts, generator)
        # 3-4. rewards -> advantages
        trajs = [r.as_traj() for r in rollouts]
        advs, adv_diag = compute_advantages(trajs, args.adv_estimator, gamma=args.gamma, lam=args.lam,
                                            step_w=args.step_adv_w, gigpo_zscore=args.gigpo_zscore)
        samples = []
        for r, a in zip(rollouts, advs):
            for d, av in zip(r.decisions, a):
                samples.append({"prompt_ids": d["prompt_ids"], "completion_ids": d["completion_ids"],
                                "advantage": float(av)})
        # 5. update
        t_up = time.time()
        upd = policy_update(model, optimizer, scaler, samples, ref_mode=ref_mode, beta=args.beta,
                            loss_type=args.loss, clip_eps=args.clip_eps, ppo_epochs=args.ppo_epochs,
                            micro_batch=args.micro_batch_size, max_grad_norm=args.max_grad_norm,
                            temperature=args.temperature if args.temperature > 0 else 1.0,
                            amp=amp, pad_id=pad_id)
        update_time = time.time() - t_up
        decs = [d for r in rollouts for d in r.decisions]
        flat_adv = [x for a in advs for x in a]
        row = {
            "step": step, "estimator": args.adv_estimator, "reward": args.reward,
            **{k: round(v, 5) for k, v in group_return_stats(trajs).items()},
            **{k: round(v, 6) for k, v in upd.items()},
            "adv_mean": round(S.fmean(flat_adv), 5) if flat_adv else 0.0,
            "adv_std": round(S.pstdev(flat_adv), 5) if len(flat_adv) > 1 else 0.0,
            "a0_life_mean": round(S.fmean(r.a0_life() for r in rollouts), 3),
            "a0_alive_end_frac": round(sum(r.a0.is_alive for r in rollouts) / len(rollouts), 4),
            "n_rollouts": len(rollouts), "n_decisions": len(decs),
            "parse_rate": round(sum(d["parsed"] for d in decs) / max(1, len(decs)), 4),
            "gen_tokens": roll["gen_tokens"], "gen_rounds": roll["rounds"],
            "gen_time_s": round(roll["gen_time"], 2),
            "tokens_per_s": round(roll["gen_tokens"] / roll["gen_time"], 1) if roll["gen_time"] > 0 else None,
            "update_time_s": round(update_time, 2), "wall_time_s": round(time.time() - t_step, 2),
            "start_t": [sc["t"] for sc, _, _ in starts],
            "adv_diag": adv_diag,
        }
        if cuda:
            row["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        rows.append(row)
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        logger.info(f"[step {step}] R={row['return_mean']:+.3f}±{row['return_std']:.3f} "
                    f"zero_var={row['zero_var_group_frac']:.2f} life={row['a0_life_mean']:.1f} "
                    f"parse={row['parse_rate']:.2f} kl={row.get('kl', 0):.4f} loss={row['loss']:+.4f} "
                    f"dec={len(decs)} gen={row['gen_time_s']}s upd={row['update_time_s']}s "
                    f"wall={row['wall_time_s']}s")
        if step % args.save_steps == 0 or step == args.max_steps:
            path = save_adapter(model, args.output_dir, step, args.save_total_limit)
            logger.info(f"saved {path}")
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--init-adapter", default=None, help="LoRA to start from (e.g. checkpoints/sft_dagger2); "
                   "omit for a fresh LoRA")
    p.add_argument("--ref-adapter", default=None, help="KL reference adapter; default = --init-adapter "
                   "(set it to the original SFT adapter when resuming from an RL checkpoint)")
    p.add_argument("--output-dir", default="checkpoints/run8")
    p.add_argument("--balance", default="v3-rc1", help="balance preset, passed through to get_balance")
    p.add_argument("--teammate-policy", default="camp")
    p.add_argument("--num-scenarios", type=int, default=200, help="start-state pool size")
    p.add_argument("--routine-frac", type=float, default=0.2)
    p.add_argument("--scenario-seed", type=int, default=42)
    p.add_argument("--batch-starts", type=int, default=4, help="B start states per step")
    p.add_argument("--group-size", type=int, default=8, help="G rollouts per start state")
    p.add_argument("--horizon", type=int, default=30, help="H game turns per rollout")
    p.add_argument("--prefix-actions", type=int, default=5, help="K: re-plan every K A0 turns")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--max-prompt-tokens", type=int, default=0, help="left-truncate prompts (0 = off)")
    p.add_argument("--gen-batch-size", type=int, default=32)
    p.add_argument("--reward", choices=["graded", "env"], default="graded")
    p.add_argument("--adv-estimator", choices=["grpo", "gigpo", "gagpo"], default="gagpo")
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lam", type=float, default=0.8)
    p.add_argument("--step-adv-w", type=float, default=1.0, help="gigpo step-advantage weight")
    p.add_argument("--gigpo-zscore", action="store_true", help="gigpo: z-score inside anchor clusters")
    p.add_argument("--loss", choices=["ppo", "reinforce"], default="ppo")
    p.add_argument("--ppo-epochs", type=int, default=1)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--beta", type=float, default=0.01, help="KL coefficient to the reference policy")
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--save-steps", type=int, default=10)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--no-grad-ckpt", action="store_true")
    p.add_argument("--no-amp", action="store_true", help="disable fp16 autocast on CUDA")
    p.add_argument("--allow-cpu", action="store_true", help="run without CUDA (tests only)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    logging.getLogger("survivecity_v2_env").setLevel(logging.WARNING)
    args = parse_args(argv)
    cuda = torch.cuda.is_available()
    logger.info(f"Device: {'cuda ' + torch.cuda.get_device_name(0) if cuda else 'cpu'}")
    if not cuda and not args.allow_cpu:
        raise SystemExit("CUDA not available; refusing to train on CPU (pass --allow-cpu for tests)")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                torch_dtype=torch.float16 if cuda else torch.float32)
    if cuda:
        base.to("cuda")
    model, ref_mode = prepare_policy(base, init_adapter=args.init_adapter, ref_adapter=args.ref_adapter,
                                     lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                                     grad_ckpt=not args.no_grad_ckpt)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"policy: init={args.init_adapter or 'fresh LoRA'} ref={ref_mode}"
                f"({args.ref_adapter or args.init_adapter or 'base'}) trainable={n_train / 1e6:.1f}M "
                f"estimator={args.adv_estimator} reward={args.reward} B={args.batch_starts} "
                f"G={args.group_size} H={args.horizon} K={args.prefix_actions}")
    train(args, model, tok, ref_mode)


if __name__ == "__main__":
    main()
