"""GiGPO — Group-in-Group Policy Optimization math helpers.

Phase 2 of `.planning2/11_RESEARCH_FINDINGS_AND_REVISED_PLAN.md`.

This module is a faithful port of the advantage-side helpers in verl-agent's
`gigpo/core_gigpo.py` (Apache 2.0, Feng et al., arxiv 2505.10978), adapted to
our TRL 0.15.2 + GRPOTrainer stack. The PPO clipped-loss part is left to TRL
— we only override the advantage tensor.

What GiGPO adds on top of GRPO:

  1. Build an ANCHOR-STATE CLUSTERING over all generations in a reward-fn
     call: two (prompt, completion) pairs whose first-action observation
     matches end up in the same cluster.
  2. Compute a STEP-LEVEL ADVANTAGE: within each cluster, normalise the
     first-action raw reward against the cluster mean (or mean/std).
  3. The final per-prompt advantage given to the PPO update is then
        A = A_episode + step_advantage_w * A_step
     where A_episode is the standard GRPO within-prompt-group advantage.

Why this fits SurviveCity v2 without rewriting the env: the env already
exposes `metadata.raw_reward` for every step, and our reward function
already records the first-action raw reward (`model_step_raws[0]` in
`train.py`). The only new thing we extract is a coarse "anchor key" for
the state the agent was in when it took that action.

The episode-level advantage stays in TRL's hands. We just add step-level
on top — see `training/gigpo_trainer.GiGPOTrainer` for the integration
point.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable, Optional, Sequence

import torch

logger = logging.getLogger("survivecity_v2.gigpo")


# ---------------------------------------------------------------------------
# Anchor key — what makes two (prompt, completion) states "the same"
# ---------------------------------------------------------------------------

def anchor_key_for_agent0(obs: dict) -> str:
    """Build a hashable key describing agent-0's state for step-level grouping.

    Design (revised 2026-06 — see memory `gigpo-anchor-inert-finding`):

    The original key was too COARSE (2x2 quadrant, hunger//4) AND keyed on a
    step bucket. Combined with the reward fn snapshotting the anchor only at
    env.reset() (a constant spawn state), every generation in a GRPO group fell
    into ONE cluster and GiGPO degenerated to GRPO. A CPU probe over the real
    env showed that an *exploring* policy diverges to 4-6 distinct fine states
    by model-turn 4, but the coarse key collapsed them to 2-3. So we refine:

        - agent-0 EXACT (row, col)   — captures the movement that actually
          differs between trajectories in the early game.
        - HP (0-3)
        - hunger half-bucket (//2)   — finer than the old //4.
        - thirst half-bucket (//2)
        - inside-safehouse boolean
        - day/night phase

    Crucially there is **no step bucket**. Two trajectories that visit the same
    cell with the same survival stats at *different* timesteps now land in the
    same step-level group — that cross-time matching is the core of GiGPO's
    "anchor state grouping". Genuinely different situations are still separated
    because hunger/thirst (which monotonically rise with time) are in the key.

    This key is only meaningful when the reward fn captures it at EACH model
    step's pre-action state (not once at reset) and the run uses
    `--prefix-actions K>1`. See `compose_gigpo_advantages_multistep`.

    Args:
        obs: env observation dict (output of env.reset() or env.step()).

    Returns:
        A short hashable string.
    """
    # Pull agent-0's masked state out of the obs.agents list.
    a0 = None
    for a in obs.get("agents", []):
        if a.get("agent_id") == 0:
            a0 = a
            break
    if a0 is None:
        # Defensive: if A0 isn't in obs (dead, mid-respawn), bucket as a sentinel.
        return "NO_A0"

    row, col = a0.get("row", 0), a0.get("col", 0)
    hp = a0.get("hp", 3)
    hunger_bucket = min(a0.get("hunger", 0) // 2, 7)
    thirst_bucket = min(a0.get("thirst", 0) // 2, 7)
    in_safe = (6 <= row <= 8) and (6 <= col <= 8)
    metadata = obs.get("metadata") or {}
    day_phase = metadata.get("day_phase", "day")

    return (
        f"r={row}|c={col}|hp={hp}"
        f"|h={hunger_bucket}|t={thirst_bucket}|s={int(in_safe)}|d={day_phase[0]}"
    )


# ---------------------------------------------------------------------------
# Step-level grouping — cluster generations by anchor key inside each prompt
# ---------------------------------------------------------------------------

def build_step_group(
    anchor_keys: Sequence[str],
    prompt_index: Sequence[int],
) -> list[str]:
    """Cluster (prompt, completion) entries by (prompt_id, anchor_key).

    GiGPO's "step-level group" is the set of generations that BOTH share a
    prompt AND visited the same anchor state. We use a single composite
    key — concat of prompt_index and anchor_key — to identify the cluster
    each entry belongs to. The exact cluster identity (a UUID in the
    paper, a string here) doesn't matter; only the partition does.

    Args:
        anchor_keys: per-entry anchor-state string (output of
            `anchor_key_for_agent0`).
        prompt_index: per-entry prompt group id (the scenario_id /
            prompt-row id from GRPO's view).

    Returns:
        List of cluster ids, one per input entry, in the same order.

    Cluster size summary is logged (avg cluster size, single-member fraction).
    """
    assert len(anchor_keys) == len(prompt_index), (
        f"anchor_keys ({len(anchor_keys)}) and prompt_index "
        f"({len(prompt_index)}) must align"
    )
    cluster_ids: list[str] = [
        f"p={p}|a={a}" for p, a in zip(prompt_index, anchor_keys)
    ]
    return cluster_ids


def cluster_size_summary(cluster_ids: Sequence[str]) -> dict:
    """Return diagnostic stats on the clustering — for [metrics] logs."""
    counts = defaultdict(int)
    for cid in cluster_ids:
        counts[cid] += 1
    sizes = sorted(counts.values(), reverse=True)
    n = len(sizes)
    if n == 0:
        return {"n_clusters": 0, "mean_size": 0.0, "max_size": 0, "singletons": 0}
    singletons = sum(1 for s in sizes if s == 1)
    return {
        "n_clusters": n,
        "mean_size": round(sum(sizes) / n, 2),
        "max_size": sizes[0],
        "singletons": singletons,
        "singleton_frac": round(singletons / n, 2),
    }


# ---------------------------------------------------------------------------
# Step-level reward normalisation — within-cluster mean/std subtract
# ---------------------------------------------------------------------------

def step_norm_reward(
    step_rewards: torch.Tensor,
    cluster_ids: Sequence[str],
    remove_std: bool = True,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute step-level advantages by normalising step rewards per cluster.

    Mirrors `step_norm_reward` from verl-agent's `gigpo/core_gigpo.py`.

    For each cluster (set of entries sharing the same cluster_id):
      - if mode == mean_norm:        adv[i] = r[i] - cluster_mean
      - if mode == mean_std_norm:    adv[i] = (r[i] - cluster_mean) / (cluster_std + eps)

    Singleton clusters (one member) get advantage 0 — they have no peer to
    compare against. This is the same convention as the upstream code.

    Args:
        step_rewards: tensor of shape (N,) — one float per (prompt, completion).
        cluster_ids: list of N strings — the cluster each entry belongs to.
        remove_std: if True, only subtract mean ("mean_norm" mode); else
            standardise ("mean_std_norm" mode). Paper default is mean_norm.
        epsilon: numerical-stability term added to std denom.

    Returns:
        Tensor of shape (N,) — the step-level advantages.
    """
    if not isinstance(step_rewards, torch.Tensor):
        step_rewards = torch.tensor(step_rewards, dtype=torch.float32)
    assert step_rewards.dim() == 1, f"expected 1D, got shape {step_rewards.shape}"
    assert len(cluster_ids) == step_rewards.shape[0], (
        f"cluster_ids ({len(cluster_ids)}) must match step_rewards "
        f"({step_rewards.shape[0]})"
    )

    # Group indices by cluster id.
    groups: dict[str, list[int]] = defaultdict(list)
    for i, cid in enumerate(cluster_ids):
        groups[cid].append(i)

    out = torch.zeros_like(step_rewards)
    for cid, idxs in groups.items():
        if len(idxs) <= 1:
            # Singleton — advantage is 0. No comparison group.
            continue
        members = step_rewards[idxs]
        m = members.mean()
        if remove_std:
            out[idxs] = members - m
        else:
            s = members.std(unbiased=False)
            out[idxs] = (members - m) / (s + epsilon)
    return out


# ---------------------------------------------------------------------------
# Public entry point — combine episode + step advantages
# ---------------------------------------------------------------------------

def compose_gigpo_advantages(
    episode_advantages: torch.Tensor,
    step_rewards: torch.Tensor,
    anchor_keys: Sequence[str],
    prompt_index: Sequence[int],
    step_advantage_w: float = 1.0,
    remove_std: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Combine GRPO's episode advantage with the GiGPO step advantage.

    Args:
        episode_advantages: shape (N,) — what TRL's GRPOTrainer already
            computes (the within-prompt-group normalised reward).
        step_rewards: shape (N,) — per-generation first-action raw reward.
        anchor_keys: list of N strings — see `anchor_key_for_agent0`.
        prompt_index: list of N ints — which GRPO prompt group each
            generation came from.
        step_advantage_w: weight on the step term. Paper default 1.0,
            insensitive in [0.4, 1.2].
        remove_std: passed to `step_norm_reward`. Paper default True.

    Returns:
        (combined_advantages, diagnostics) where:
          combined_advantages: shape (N,) — final advantage tensor for PPO.
          diagnostics: dict of cluster stats for logging.
    """
    cluster_ids = build_step_group(anchor_keys, prompt_index)
    step_adv = step_norm_reward(
        step_rewards, cluster_ids, remove_std=remove_std,
    )
    # Match dtype/device of the episode advantages so the sum is well-typed.
    step_adv = step_adv.to(
        dtype=episode_advantages.dtype, device=episode_advantages.device,
    )
    combined = episode_advantages + step_advantage_w * step_adv

    diag = cluster_size_summary(cluster_ids)
    diag["step_adv_mean"] = float(step_adv.mean().item())
    diag["step_adv_std"] = float(step_adv.std(unbiased=False).item()) if step_adv.numel() > 1 else 0.0
    diag["step_advantage_w"] = step_advantage_w
    return combined, diag


# ---------------------------------------------------------------------------
# Multi-step entry point — the one that actually implements GiGPO's mechanism
# ---------------------------------------------------------------------------

def compose_gigpo_advantages_multistep(
    episode_advantages: torch.Tensor,
    per_gen_steps: Sequence[Sequence[tuple]],
    prompt_index: Sequence[int],
    step_advantage_w: float = 1.0,
    remove_std: bool = True,
    step_reduce: str = "sum",
) -> tuple[torch.Tensor, dict]:
    """GiGPO step-level advantage over MULTIPLE model steps per generation.

    This is the faithful version of GiGPO's "anchor state grouping": every
    (generation, model-step) pair is one entry, ALL entries are pooled, and
    entries that share a (prompt, anchor) land in the same step-level group —
    so an action taken from state X at step 2 of trajectory i is compared
    against an action taken from the *same* state X at step 4 of trajectory j.
    That cross-time, cross-trajectory comparison is the whole point, and it is
    impossible with the single-anchor-at-reset shape of
    `compose_gigpo_advantages` (which degenerates to GRPO when every
    generation in a group shares the same reset anchor).

    The per-(gen,step) step advantages are then reduced back to ONE scalar per
    generation (our completions carry a single GRPO advantage each, since the
    whole K-action plan is one completion), and added to the episode advantage.

    Args:
        episode_advantages: shape (N,) — TRL's within-group normalised reward.
        per_gen_steps: length-N sequence; entry i is a list of
            (anchor_key, step_reward) tuples, one per model action that
            generation i actually took. May be empty (generation errored or
            never got to act) — such a generation gets step advantage 0.
        prompt_index: length-N — which GRPO prompt group each generation is in.
        step_advantage_w: weight on the aggregated step term. Paper default 1.0.
        remove_std: passed to `step_norm_reward` (mean-subtract vs z-score).
        step_reduce: how to collapse a generation's per-step advantages into a
            single scalar — "sum" (default; more steps → more signal) or
            "mean" (length-normalised, stops long survivors dominating).

    Returns:
        (combined_advantages, diagnostics).
    """
    n = episode_advantages.shape[0]
    assert len(per_gen_steps) == n, (
        f"per_gen_steps ({len(per_gen_steps)}) must match episode_advantages ({n})"
    )
    assert len(prompt_index) == n, (
        f"prompt_index ({len(prompt_index)}) must match episode_advantages ({n})"
    )
    assert step_reduce in ("sum", "mean"), f"bad step_reduce={step_reduce}"

    # Flatten every (generation, model-step) into a pooled entry list.
    flat_anchor: list[str] = []
    flat_reward: list[float] = []
    flat_owner: list[int] = []   # which generation each flat entry belongs to
    for gi, steps in enumerate(per_gen_steps):
        for entry in steps:
            anchor, step_r = entry
            flat_anchor.append(anchor)
            flat_reward.append(float(step_r))
            flat_owner.append(gi)

    per_gen_step_adv = torch.zeros(
        n, dtype=episode_advantages.dtype, device=episode_advantages.device,
    )

    if flat_anchor:
        flat_prompt = [prompt_index[o] for o in flat_owner]
        cluster_ids = build_step_group(flat_anchor, flat_prompt)
        flat_step_adv = step_norm_reward(
            torch.tensor(flat_reward, dtype=torch.float32),
            cluster_ids,
            remove_std=remove_std,
        ).to(dtype=episode_advantages.dtype, device=episode_advantages.device)

        # Reduce per-(gen,step) advantages back to one scalar per generation.
        counts = torch.zeros(n, device=episode_advantages.device)
        for adv_val, gi in zip(flat_step_adv, flat_owner):
            per_gen_step_adv[gi] += adv_val
            counts[gi] += 1
        if step_reduce == "mean":
            per_gen_step_adv = per_gen_step_adv / counts.clamp(min=1.0)

        diag = cluster_size_summary(cluster_ids)
        diag["n_step_entries"] = len(flat_anchor)
        diag["step_adv_std"] = (
            float(flat_step_adv.std(unbiased=False).item())
            if flat_step_adv.numel() > 1 else 0.0
        )
    else:
        diag = cluster_size_summary([])
        diag["n_step_entries"] = 0
        diag["step_adv_std"] = 0.0

    combined = episode_advantages + step_advantage_w * per_gen_step_adv
    diag["step_adv_mean"] = float(per_gen_step_adv.mean().item())
    diag["step_advantage_w"] = step_advantage_w
    diag["step_reduce"] = step_reduce
    return combined, diag
