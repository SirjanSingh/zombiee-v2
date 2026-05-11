"""Unit tests for training/gigpo.py — math only, CPU-only, no ML deps.

These exist because PPO-family math is notorious for silent failures:
shape mismatches, off-by-one cluster assignments, std=0 divide-by-zero,
etc. Running this test suite (`pytest tests/test_gigpo.py`) before any
DGX run takes <2 seconds and catches the most likely Phase 2 regressions.
"""

from __future__ import annotations

import torch

from training.gigpo import (
    anchor_key_for_agent0,
    build_step_group,
    cluster_size_summary,
    compose_gigpo_advantages,
    step_norm_reward,
)


# ---------------------------------------------------------------------------
# anchor_key_for_agent0
# ---------------------------------------------------------------------------

def _mk_obs(
    step=0, row=7, col=7, hp=3, hunger=0, thirst=0, day_phase="day",
):
    """Tiny obs dict shaped like SurviveCityV2Env's output."""
    return {
        "step_count": step,
        "agents": [{
            "agent_id": 0,
            "row": row,
            "col": col,
            "hp": hp,
            "hunger": hunger,
            "thirst": thirst,
            "is_alive": True,
        }],
        "metadata": {"day_phase": day_phase},
    }


def test_anchor_key_is_deterministic():
    obs = _mk_obs()
    assert anchor_key_for_agent0(obs) == anchor_key_for_agent0(obs)


def test_anchor_key_differs_on_step_bucket_change():
    # Step 0 and step 4 same bucket; step 5 different bucket.
    assert anchor_key_for_agent0(_mk_obs(step=0)) == anchor_key_for_agent0(_mk_obs(step=4))
    assert anchor_key_for_agent0(_mk_obs(step=0)) != anchor_key_for_agent0(_mk_obs(step=5))


def test_anchor_key_differs_on_quadrant_change():
    # Quadrant is row//8, col//8 — rows 0-7 vs 8-14.
    k_top = anchor_key_for_agent0(_mk_obs(row=3, col=3))
    k_bot = anchor_key_for_agent0(_mk_obs(row=10, col=3))
    assert k_top != k_bot


def test_anchor_key_handles_missing_agent0():
    # Defensive: obs with no A0 agent (mid-respawn, dead) returns a sentinel.
    obs = {"step_count": 0, "agents": [], "metadata": {}}
    assert anchor_key_for_agent0(obs) == "NO_A0"


def test_anchor_key_buckets_hunger():
    # Hunger 0, 3 → bucket 0. Hunger 4, 7 → bucket 1. Hunger 12, 15 → bucket 3.
    assert anchor_key_for_agent0(_mk_obs(hunger=0)) == anchor_key_for_agent0(_mk_obs(hunger=3))
    assert anchor_key_for_agent0(_mk_obs(hunger=4)) == anchor_key_for_agent0(_mk_obs(hunger=7))
    assert anchor_key_for_agent0(_mk_obs(hunger=0)) != anchor_key_for_agent0(_mk_obs(hunger=4))


# ---------------------------------------------------------------------------
# build_step_group
# ---------------------------------------------------------------------------

def test_build_step_group_separates_by_prompt():
    # Same anchor key, different prompts → different clusters.
    anchors = ["A", "A", "A"]
    prompts = [0, 1, 2]
    cids = build_step_group(anchors, prompts)
    assert len(set(cids)) == 3


def test_build_step_group_clusters_by_anchor_within_prompt():
    # Within prompt 0: two with anchor A, one with anchor B → 2 clusters.
    anchors = ["A", "A", "B"]
    prompts = [0, 0, 0]
    cids = build_step_group(anchors, prompts)
    assert cids[0] == cids[1]
    assert cids[0] != cids[2]


def test_build_step_group_length_matches_input():
    anchors = ["A"] * 8
    prompts = [0] * 8
    cids = build_step_group(anchors, prompts)
    assert len(cids) == 8


# ---------------------------------------------------------------------------
# step_norm_reward
# ---------------------------------------------------------------------------

def test_step_norm_singleton_cluster_gets_zero_advantage():
    # Only one entry in its cluster → no peer comparison → adv = 0.
    rewards = torch.tensor([1.5, 0.5, 2.0])
    cids = ["unique_a", "unique_b", "unique_c"]
    adv = step_norm_reward(rewards, cids, remove_std=True)
    assert torch.allclose(adv, torch.zeros_like(adv))


def test_step_norm_mean_subtract_is_correct():
    # Cluster of [1.0, 3.0]: mean=2.0 → advantages [-1.0, +1.0].
    rewards = torch.tensor([1.0, 3.0])
    cids = ["c", "c"]
    adv = step_norm_reward(rewards, cids, remove_std=True)
    assert torch.allclose(adv, torch.tensor([-1.0, 1.0]))


def test_step_norm_mean_std_is_zscore():
    rewards = torch.tensor([1.0, 2.0, 3.0])
    cids = ["c", "c", "c"]
    adv = step_norm_reward(rewards, cids, remove_std=False)
    # mean=2, std=sqrt(2/3); z-scores summing to 0; symmetric around 0.
    assert abs(float(adv.sum().item())) < 1e-5
    assert adv[0] < adv[1] < adv[2]


def test_step_norm_handles_mixed_singletons_and_clusters():
    # cluster X has 3 members, cluster Y is a singleton.
    rewards = torch.tensor([10.0, 0.0, 5.0, 99.0])
    cids = ["X", "X", "X", "Y"]
    adv = step_norm_reward(rewards, cids, remove_std=True)
    # X has mean 5 → advs [+5, -5, 0]; Y singleton → 0.
    assert torch.allclose(adv, torch.tensor([5.0, -5.0, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# cluster_size_summary
# ---------------------------------------------------------------------------

def test_cluster_size_summary_basic():
    cids = ["a", "a", "b", "c", "c", "c"]
    s = cluster_size_summary(cids)
    assert s["n_clusters"] == 3
    assert s["mean_size"] == 2.0
    assert s["max_size"] == 3
    assert s["singletons"] == 1


def test_cluster_size_summary_empty():
    s = cluster_size_summary([])
    assert s["n_clusters"] == 0
    assert s["mean_size"] == 0.0


# ---------------------------------------------------------------------------
# compose_gigpo_advantages — full integration
# ---------------------------------------------------------------------------

def test_compose_gigpo_zero_weight_is_pure_episode():
    # step_advantage_w=0 → output == episode_advantages.
    ep_adv = torch.tensor([0.5, -0.3, 0.8, -1.0])
    step_r = torch.tensor([1.0, 2.0, 3.0, 4.0])
    anchors = ["A", "B", "A", "B"]
    prompts = [0, 0, 0, 0]
    out, _diag = compose_gigpo_advantages(
        ep_adv, step_r, anchors, prompts, step_advantage_w=0.0,
    )
    assert torch.allclose(out, ep_adv)


def test_compose_gigpo_adds_step_advantage():
    # All 4 generations are in cluster (prompt=0, anchor=A) — step_mean = 2.5.
    # step advantages = [-1.5, -0.5, +0.5, +1.5].
    ep_adv = torch.zeros(4)
    step_r = torch.tensor([1.0, 2.0, 3.0, 4.0])
    anchors = ["A"] * 4
    prompts = [0] * 4
    out, diag = compose_gigpo_advantages(
        ep_adv, step_r, anchors, prompts, step_advantage_w=1.0,
    )
    assert torch.allclose(out, torch.tensor([-1.5, -0.5, 0.5, 1.5]))
    assert diag["n_clusters"] == 1
    assert diag["mean_size"] == 4.0


def test_compose_gigpo_diagnostics_shape():
    ep_adv = torch.zeros(8)
    step_r = torch.zeros(8)
    anchors = ["A"] * 4 + ["B"] * 4
    prompts = [0] * 8
    _, diag = compose_gigpo_advantages(
        ep_adv, step_r, anchors, prompts, step_advantage_w=1.0,
    )
    assert diag["n_clusters"] == 2
    assert diag["mean_size"] == 4.0
    assert "step_adv_mean" in diag
    assert "step_advantage_w" in diag


def test_compose_gigpo_dtype_device_preserved():
    ep_adv = torch.tensor([0.0, 0.0], dtype=torch.float64)
    step_r = torch.tensor([1.0, 3.0])
    out, _ = compose_gigpo_advantages(
        ep_adv, step_r, ["A", "A"], [0, 0], step_advantage_w=1.0,
    )
    assert out.dtype == torch.float64
