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
    """Build a coarse, hashable key describing agent-0's state.

    The full observation text is too unique — every grid configuration is
    a slightly different string and no two trajectories cluster. We
    discretise into the dimensions that actually matter for credit
    assignment:

        - step bucket (5-step granularity — fine enough to separate "before
          first wave" from "after first wave" without exploding cardinality)
        - agent-0 quadrant on the 15x15 grid (2x2 quadrants)
        - HP (already 0-3)
        - hunger bucket (4 buckets of size 4: 0-3, 4-7, 8-11, 12-15)
        - thirst bucket (same scheme)
        - inside-safehouse boolean
        - day/night phase

    Total cardinality ≤ 20 * 4 * 4 * 4 * 4 * 2 * 2 ≈ 20k. Empirically the
    cluster average we want is ~2-3 (paper reports 2.4 on ALFWorld).

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

    step = obs.get("step_count", 0)
    step_bucket = step // 5
    row, col = a0.get("row", 0), a0.get("col", 0)
    quadrant = (row // 8, col // 8)  # 0..1, 0..1 → 4 quadrants
    hp = a0.get("hp", 3)
    hunger_bucket = min(a0.get("hunger", 0) // 4, 3)
    thirst_bucket = min(a0.get("thirst", 0) // 4, 3)
    in_safe = (6 <= row <= 8) and (6 <= col <= 8)
    metadata = obs.get("metadata") or {}
    day_phase = metadata.get("day_phase", "day")

    return (
        f"sb={step_bucket}|q={quadrant[0]}{quadrant[1]}|hp={hp}"
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
