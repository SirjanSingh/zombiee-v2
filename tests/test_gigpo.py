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
    compose_gigpo_advantages_multistep,
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


def test_anchor_key_ignores_step():
    # Revised design: NO step bucket. The same position/stats at different
    # timesteps yield the SAME key — this is what enables GiGPO's cross-time
    # anchor-state matching (same state visited at different turns clusters).
    assert anchor_key_for_agent0(_mk_obs(step=0)) == anchor_key_for_agent0(_mk_obs(step=9))


def test_anchor_key_differs_on_position():
    # Exact position now: even a one-cell move changes the key.
    assert anchor_key_for_agent0(_mk_obs(row=3, col=3)) != anchor_key_for_agent0(_mk_obs(row=10, col=3))
    assert anchor_key_for_agent0(_mk_obs(row=3, col=3)) != anchor_key_for_agent0(_mk_obs(row=3, col=4))


def test_anchor_key_handles_missing_agent0():
    # Defensive: obs with no A0 agent (mid-respawn, dead) returns a sentinel.
    obs = {"step_count": 0, "agents": [], "metadata": {}}
    assert anchor_key_for_agent0(obs) == "NO_A0"


def test_anchor_key_buckets_hunger():
    # Revised: hunger //2 buckets. 0,1 → bucket 0. 2,3 → bucket 1.
    assert anchor_key_for_agent0(_mk_obs(hunger=0)) == anchor_key_for_agent0(_mk_obs(hunger=1))
    assert anchor_key_for_agent0(_mk_obs(hunger=2)) == anchor_key_for_agent0(_mk_obs(hunger=3))
    assert anchor_key_for_agent0(_mk_obs(hunger=0)) != anchor_key_for_agent0(_mk_obs(hunger=2))


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


# ---------------------------------------------------------------------------
# compose_gigpo_advantages_multistep — the real GiGPO mechanism
# ---------------------------------------------------------------------------

def test_multistep_all_empty_is_pure_episode():
    # No model steps recorded (e.g. errored gens) → step advantage 0.
    ep = torch.tensor([0.5, -0.3])
    out, diag = compose_gigpo_advantages_multistep(
        ep, [[], []], [0, 0], step_advantage_w=1.0,
    )
    assert torch.allclose(out, ep)
    assert diag["n_step_entries"] == 0


def test_multistep_zero_weight_is_pure_episode():
    ep = torch.tensor([0.5, -0.3])
    per_gen = [[("X", 1.0)], [("X", 5.0)]]
    out, _ = compose_gigpo_advantages_multistep(
        ep, per_gen, [0, 0], step_advantage_w=0.0,
    )
    assert torch.allclose(out, ep)


def test_multistep_cross_time_clustering():
    # The core property: anchor X appears at step 0 of gen0 AND step 1 of gen1.
    # They must land in the SAME step-level cluster despite different turns.
    #   X cluster rewards [1.0(gen0), 3.0(gen1)] → mean 2 → advs [-1, +1]
    #   Y cluster rewards [2.0(gen0), 4.0(gen1)] → mean 3 → advs [-1, +1]
    #   per-gen sum: gen0 = -1 + -1 = -2 ; gen1 = +1 + +1 = +2
    ep = torch.zeros(2)
    per_gen = [[("X", 1.0), ("Y", 2.0)], [("Y", 4.0), ("X", 3.0)]]
    out, diag = compose_gigpo_advantages_multistep(
        ep, per_gen, [0, 0], step_advantage_w=1.0,
    )
    assert torch.allclose(out, torch.tensor([-2.0, 2.0]))
    assert diag["n_clusters"] == 2
    assert diag["n_step_entries"] == 4
    assert diag["mean_size"] == 2.0


def test_multistep_different_prompts_dont_cluster():
    # Same anchor X but different GRPO prompts → singletons → 0 advantage.
    ep = torch.zeros(2)
    per_gen = [[("X", 1.0)], [("X", 5.0)]]
    out, _ = compose_gigpo_advantages_multistep(
        ep, per_gen, [0, 1], step_advantage_w=1.0,
    )
    assert torch.allclose(out, torch.zeros(2))


def test_multistep_sum_vs_mean_reduce():
    # gen0 contributes to BOTH the X and Y clusters (2 steps); gens 1,2 one each.
    #   X cluster [0.0(g0), 4.0(g1)] → mean 2 → [-2, +2]
    #   Y cluster [0.0(g0), 4.0(g2)] → mean 2 → [-2, +2]
    #   gen0 per-step advs = [-2(X), -2(Y)] ; sum=-4, mean=-2
    ep = torch.zeros(3)
    per_gen = [[("X", 0.0), ("Y", 0.0)], [("X", 4.0)], [("Y", 4.0)]]
    out_sum, _ = compose_gigpo_advantages_multistep(
        ep, per_gen, [0, 0, 0], step_advantage_w=1.0, step_reduce="sum",
    )
    out_mean, _ = compose_gigpo_advantages_multistep(
        ep, per_gen, [0, 0, 0], step_advantage_w=1.0, step_reduce="mean",
    )
    assert torch.allclose(out_sum, torch.tensor([-4.0, 2.0, 2.0]))
    assert torch.allclose(out_mean, torch.tensor([-2.0, 2.0, 2.0]))


def test_multistep_dtype_preserved():
    ep = torch.zeros(2, dtype=torch.float64)
    per_gen = [[("X", 1.0)], [("X", 3.0)]]
    out, _ = compose_gigpo_advantages_multistep(
        ep, per_gen, [0, 0], step_advantage_w=1.0,
    )
    assert out.dtype == torch.float64
