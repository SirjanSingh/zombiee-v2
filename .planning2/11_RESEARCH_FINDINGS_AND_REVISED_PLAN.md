# 11 — Research findings + revised plan (supersedes plan 10)

**Status:** plan, ready to execute Phase 1 immediately
**Author:** Claude (research synthesis)
**Created:** 2026-05-11
**Supersedes:** `.planning2/10_PPO_TRAJECTORY.md` (the PPO plan was over-engineered — see §1)

---

## TL;DR — the honest answer

After spending 80 hours on GRPO with disappointing results, I went looking for
**published recipes that actually work for our problem class** (Qwen2.5-3B,
multi-step environment, LoRA, ~60-step episodes). What I found:

1. **PPO from scratch is the wrong move** for our setup. The published winner
   for "LLM agent in multi-step environment with 3B model" is
   [**GiGPO (May 2025)**](https://arxiv.org/abs/2505.10978) — Group-in-Group
   Policy Optimization. It is GRPO + a step-level advantage from anchor-state
   grouping, **no critic, no value head, no PPO**. It beats GRPO by **>12% on
   ALFWorld** and beats critic-based PPO on the same benchmark.
2. **Our hyperparameters are miscalibrated by 5-10×** across multiple knobs.
   The published GiGPO recipe for Qwen2.5-3B on ALFWorld uses LR=3e-6 (we use
   1e-5), KL β=0.01 (we use 0.04), 150 epochs (we run 40-100), and KL type
   `low_var_kl` (we use TRL default, the high-variance approximation). Even
   *before* an algorithm change, fixing these is a likely big win.
3. **The "Echo Trap" failure pattern** documented in
   [RAGEN (April 2025)](https://arxiv.org/abs/2504.20073) — entropy collapse,
   reward variance cliff, repetitive deterministic outputs, gradient spikes —
   matches our runs 1-4 exactly. RAGEN's remedy (StarPO-S: top-25%
   trajectory filtering + DAPO clip-higher + KL removal) gives us a real,
   tested escalation path if Phase 1+2 fall short.
4. **SFT bootstrap is highly recommended.**
   [Sordoni et al. 2025](https://arxiv.org/html/2507.04103) ("How to Train
   Your LLM Web Agent: A Statistical Diagnosis") found "pure RL from
   scratch particularly struggles due to sparse rewards and unstable
   learning dynamics" and recommend SFT-then-RL. We have the
   forage_heuristic_action which gives us an instant expert demonstrator —
   we should use it.

The previous PPO plan would have been **1-2 weeks of new code**. The plan
below is **~3 days** for Phase 1+2 with each phase having a clear
abort/escalate decision.

---

## 1. Why PPO with value head was wrong

In my previous response I said "PPO with a value head is the right algorithm".
That was reasoning from first principles — credit assignment requires a value
function, therefore PPO. After reading the literature, that reasoning ignores
two facts:

1. **TRL's `PPOTrainer` is RLHF-shaped, not trajectory-RL-shaped.** Multi-step
   PPO on top of an LLM in a real env requires writing the trainer loop from
   scratch (logprob roundtrips, GAE across env steps, value-head + LoRA
   interaction in 4-bit). ~800 lines of new ML code with many known sharp
   edges.
2. **You can get fine-grained credit assignment WITHOUT a critic.** GiGPO's
   contribution is showing this. By grouping trajectories that visited the
   "same" state and using sibling rewards as a baseline, you get per-step
   advantages from group statistics alone. On the published benchmark
   closest to our setup (ALFWorld, multi-step, sparse reward), GiGPO beats
   PPO+critic.

PPO is not retired forever — it's our Phase 4 fallback. But starting there
ignores cheaper wins.

---

## 2. Evidence base

| Source | Date | Why it matters for us | Headline finding |
|---|---|---|---|
| [GiGPO](https://arxiv.org/abs/2505.10978) — *Feng et al. NeurIPS 2025* | May 2025 | Qwen2.5-3B + multi-step + LoRA + sparse reward — **same stack as us** | +12% over GRPO on ALFWorld via step-level advantage from anchor-state grouping. No critic. |
| [RAGEN / StarPO-S](https://arxiv.org/abs/2504.20073) — *NEU MLL* | Apr 2025 | Documents the EXACT failure pattern we have | "Echo Trap" = entropy collapse + reward-variance cliff + gradient spikes. Fixes: trajectory filtering + clip-higher + critic. |
| [How to Train Your LLM Web Agent](https://arxiv.org/html/2507.04103) | Jul 2025 | Statistical sweep of LLM-agent RL recipes | Pure RL "struggles particularly"; SFT-then-RL wins. LR 1e-6, temperature 0.25. |
| [DAPO](https://arxiv.org/pdf/2503.14476) | Mar 2025 | Source of "clip-higher" used by StarPO-S | Asymmetric clipping (ε_high > ε_low) prevents entropy collapse in PPO/GRPO. |
| [ArCHer](https://arxiv.org/abs/2402.19446) | Feb 2024 | First multi-turn RL paper for LLM agents | Hierarchical actor-critic; 100× sample efficiency over PPO. (Heavier; we're not going there yet.) |
| [verl-agent](https://github.com/langfengq/verl-agent) (the GiGPO repo) | 2025 | Reference code, Apache 2.0 | Has runnable training scripts for Qwen2.5-3B on ALFWorld/Sokoban (LoRA + full FT). |
| [verl PPO docs](https://verl.readthedocs.io/en/latest/algo/ppo.html) | 2025 | Production-grade RL framework for LLMs | Implements PPO/GRPO/RLOO/DAPO with the exact KL estimator (`low_var_kl`) and clip-higher we want. |

---

## 3. The miscalibration table

Comparing our current `train.py` defaults vs the published GiGPO Sokoban
recipe (closest to our env: single-agent grid world, discrete actions,
sparse reward). Sokoban uses Qwen2.5-3B, the same base model as us.

| Hyperparameter | Our current (run 1-4) | GiGPO Sokoban | GiGPO ALFWorld LoRA | Status |
|---|---|---|---|---|
| Actor LR | **1e-5** | **1e-6** | **3e-6** | **3-10× too high** |
| KL β (`beta`) | **0.04** | **0.01** | not specified (default ~0.01) | **4× too high** |
| KL loss type | `approx` (TRL default) | **`low_var_kl`** | `low_var_kl` (presumed) | **Wrong estimator** |
| Group size (`num_generations`) | 8 | 8 | 8 | ✓ |
| Train batch | 4 (1 × grad_accum 4) | 16 | 32 | **2-8× too small** |
| ppo_mini_batch | n/a (TRL handles) | 64 | 256 | n/a |
| Max response tokens | 384 | 512 | 512 | **Tight** |
| Max prompt tokens | 1536 | 4096 | 2048 | OK |
| Total training steps | 40 (run 3) to 100 (run 1) | **150** | 150 | **~2× too short** |
| γ (discount) | n/a for GRPO | 0.95 | 0.95 | n/a |
| Invalid action penalty | format_bonus = +0.10 if parse_ok | **−0.10 if parse fails** | −0.10 if parse fails | **Sign is reversed in spirit** |
| LoRA rank | 32 | full FT | 64 | **Half the rank** |
| Adv estimator | `grpo` | **`gigpo`** | `gigpo` | **The whole point** |

The "Sign is reversed" entry deserves comment: we ADD 0.10 for a parseable
action. GiGPO SUBTRACTS 0.10 for an unparseable one. Mathematically the
same when parse rates are stable. **But** if the model produces a mix of
parseable and unparseable outputs in a group, a flat reward bonus is
absorbed into the GRPO mean — every group member gets the same +0.10 so it
cancels in advantage. A NEGATIVE penalty for unparseable still cancels in
the average BUT — combined with the group statistics — it pushes the
unparseable trajectory below the group mean and gives it negative
advantage. That's the actual behavioural effect we want.

---

## 4. Algorithm choice: GiGPO over PPO+critic

### 4.1 What GiGPO actually is

From [`gigpo/core_gigpo.py`](https://github.com/langfengq/verl-agent/blob/master/gigpo/core_gigpo.py)
in the verl-agent repo (Apache 2.0):

```python
def compute_gigpo_outcome_advantage(...):
    # GRPO-style episode advantage: normalize per-group trajectory return
    episode_advantages = episode_norm_reward(token_level_rewards, ...)

    # GiGPO addition #1: cluster trajectories by "anchor state" (same observation
    # seen by different trajectories in the same group → step-level peer group)
    step_group_uids = build_step_group(anchor_obs, index, enable_similarity=False)

    # GiGPO addition #2: within each step-level cluster, normalize the step
    # reward against its peer group → step-level advantage
    step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids)

    # Joint advantage = episode + step (weighted)
    scores = episode_advantages + step_advantage_w * step_advantages
    return scores, scores  # plugged into the standard PPO clipped objective
```

**The brilliance:** all you need is (a) a per-step reward, (b) a way to
identify "this step is the same situation as that step in trajectory j",
and you get per-step credit assignment from group statistics alone — no
value function.

### 4.2 Why GiGPO fits SurviveCity v2 well

In our env, the 8 GRPO generations all start from the **same seed**, so
they share an identical step-0 observation. That means step 0 has a
group of 8 peer rewards to compare against — the strongest possible
GiGPO signal. With `prefix_actions=K`, the model gets K actions before
the heuristic takes over, so the first K steps have meaningful
divergence and meaningful peer comparison. After that, all trajectories
converge to heuristic actions and contribute mostly noise — which the
step-level normalization filters out naturally (the step rewards
become similar across all peers, so step_advantage ≈ 0).

### 4.3 Anchor state key for SurviveCity v2

GiGPO needs an "anchor obs" — a hashable string per (trajectory, step)
that identifies what state the agent was in. Full observation text is
too unique (every grid state is slightly different). We need a coarser
key. Proposal:

```python
def anchor_key_for_a0(obs: dict, step: int) -> str:
    a0 = obs["agents"][0]
    quadrant = (a0["row"] // 7, a0["col"] // 7)        # 2x2 = 4 quadrants
    hp = a0["hp"]                                       # 0-3
    hunger_bucket = min(a0["hunger"] // 4, 3)           # 4 buckets
    thirst_bucket = min(a0["thirst"] // 4, 3)
    in_safe = (6 <= a0["row"] <= 8) and (6 <= a0["col"] <= 8)
    phase = obs["metadata"]["phase"]                    # "day" or "night"
    return f"{step}|{quadrant}|hp={hp}|h={hunger_bucket}|t={thirst_bucket}|s={int(in_safe)}|{phase}"
```

This gives ~200 possible keys per step. With 8 trajectories per group
and ~5 model-controlled steps, we expect average step-level group size of
~2-3 (some collisions per cluster), which is the sweet spot the paper
reports (their average cluster size on ALFWorld is 2.4).

---

## 5. Phased plan

Each phase is a **commit + a DGX run**, with a **go/no-go check** at the
end. If the phase's check passes we stop and call it a win. If it fails
we go to the next phase. The phases are ordered cheap → expensive.

### Phase 1 — Hyperparameter recalibration (1 hour to code, 1 DGX run)

**Hypothesis:** our hyperparameters are so miscalibrated that GRPO can't
learn even with everything else correct.

**Changes to `training/train.py` defaults:**
- `--lr 3e-6` (was 1e-5)
- `--beta 0.01` (was 0.04)
- `--max-steps 150` (was 15-100)
- `--grad-accum-steps 16` (was 4) → effective batch = 16
- `--max-completion-length 512` (was 384)
- `--lora-r 64 --lora-alpha 128` (was 32 / 64)
- Add `--save-steps 10` so we get a useful checkpoint trail
- Add `--invalid-action-penalty 0.10` and remove `--format-bonus`. New
  reward formula:
  ```
  composite = step1_weight * sum(model_raws) + cum0 - (0.10 if parse_failed else 0.0)
  ```
- Switch TRL's `kl_loss_type` to `low_var_kl` — requires checking
  `GRPOConfig` in `trl==0.15.2`: looks like the field is `loss_type` and
  needs `"low_var_kl"` literal. If the option isn't there in 0.15.2 we
  pin a newer TRL just for this knob, or add manual KL.

**Code changes:** ~30 lines in `train.py`. No new files.

**Go/no-go after Phase 1 (eval after 150 steps):**
- ✅ Stop if `mean_alive_at_end ≥ 1.9` (heuristic baseline + 0.3) on 30
  eval episodes.
- ❌ Continue to Phase 2 if `mean_alive_at_end < 1.9` OR if training logs
  show KL=0 across all steps (the same failure as runs 3-4).

**Time:** 1 hour to code; ~3 hours of DGX time (150 steps × ~2 min/step).

### Phase 2 — Switch to GiGPO (2-3 days to code, 1 DGX run)

**Hypothesis:** vanilla GRPO can't do step-level credit assignment even
with good hyperparameters. GiGPO's step-level advantage gives the
gradient that GRPO can't.

**What we build:**
- `training/gigpo.py` (NEW): port `step_norm_reward`,
  `build_step_group`, and `compute_gigpo_outcome_advantage` from
  verl-agent (Apache 2.0; ~150 LOC). Direct copy with style tweaks.
- `training/train.py` (MODIFIED): in `create_reward_fn`, ALSO collect:
  - the FIRST-action raw reward (already collected as
    `model_step_raws[0]`),
  - the anchor key for the state where that first action was taken
    (computed once before `env.step`).
  - Return these alongside the existing scalar reward.
- Subclass `GRPOTrainer` → `GiGPOTrainer` (NEW, ~100 LOC): override
  the advantage computation to call our `compute_gigpo_outcome_advantage`
  instead of the inherited GRPO normalization. Reuse everything else.
  Key trick: hook `_get_per_token_logps` / advantage hook in TRL 0.15.2.
- `training/train.py` argparse: new flag `--adv-estimator {grpo,gigpo}`
  default `gigpo`, `--step-advantage-w 1.0`, `--anchor-mode {exact,similarity}`
  default `exact`.

**Validation gates (before any long run):**
- Gate A: in a 5-step toy run, dump the anchor-key clusters. Confirm the
  average cluster size is between 1.5 and 6 (1.0 = no clustering = pure
  GRPO; > 8 = clustering too coarse).
- Gate B: confirm advantage variance is non-zero on the first reward
  call AND `kl > 0` by step 5.

**Go/no-go after Phase 2 (eval after 150 steps):**
- ✅ Stop if `mean_alive_at_end ≥ 2.2` (heuristic + 0.6).
- ❌ Continue to Phase 3 if < 2.2.

**Time:** 2-3 days to code (most of it is the TRL subclassing dance; the
algorithm itself is 150 LOC); ~4 hours of DGX time.

### Phase 3 — SFT bootstrap from heuristic (1 day to code, 1 DGX run)

**Hypothesis:** the model can't even FORMAT correct actions reliably
when the GRPO group encounters states the heuristic handles well, so
the gradient is mostly noise. A short SFT pretrain on heuristic actions
fixes the format and gives a strong prior.

**What we build:**
- `training/build_sft_dataset.py` (NEW): run `forage_heuristic_action`
  on 500 episodes (~30k state-action pairs), save as JSONL with fields
  `{"prompt": ..., "completion": ...}`. Pick prompts at A0's turns
  only.
- Standard HF SFT: ~50 lines using `transformers.Trainer` with cross-
  entropy on the action JSON tokens only (mask the prompt). 1 epoch
  over 30k examples, batch 4, LR 5e-5. ~30 min on DGX.
- Save as a LoRA adapter; `train.py` already supports `--warmstart-from`
  to load it as the initial GRPO/GiGPO state.

**Go/no-go after Phase 3:**
- ✅ Stop if `mean_alive_at_end ≥ 2.5`.
- ❌ Continue to Phase 4 if < 2.5.

**Time:** 1 day to code; ~1.5 DGX hours (SFT) + 4 hours (GiGPO from
warmstart).

### Phase 4 — StarPO-S fallback (last resort)

Only if Phase 1+2+3 all fall short. This is the heaviest change.

**What we add to GiGPO:**
- **Trajectory filtering**: after collecting 8 generations per scenario,
  keep only the top 25% by reward variance for the gradient update.
  Drops uninformative groups.
- **DAPO clip-higher**: split the PPO clip range into
  `clip_ratio_low=0.2`, `clip_ratio_high=0.28`. Prevents entropy collapse.
  TRL 0.15.2 may not expose this; would need a custom loss override.
- **PPO with critic**: drop GiGPO, add value head, do real trajectory
  PPO. This is the plan-10 work, but only if Phase 3 fails too.

**Time:** 1 week if it gets here. Decision point: if Phase 3 fails,
revisit the env first before this. Maybe 3B is the ceiling for this
task.

---

## 6. What stays the same

The points where I'm NOT making changes vs plan 10:

- **A0-only training** stays. Multi-agent is OOD vs eval and adds
  variance. Plenty of papers (ArCHer, GiGPO) train a single-agent policy
  in a multi-agent env without issue.
- **The env, rubric, eval harness, metrics, plots** all stay. The
  per-step `raw_reward` and `cumulative_rewards[0]` exposed by the env
  already give us everything Phase 1+2+3 need.
- **4-bit LoRA, V100, Docker workflow** stay. Phase 1+2 fit
  comfortably in 32 GB. Phase 3 SFT runs on the same image.
- **noanya/zombiee-v2 Hub repo** stays as the artifact store.

---

## 7. Decisions for you to make

1. **Start with Phase 1 today, or go straight to Phase 2?**
   *My recommendation:* **Phase 1 today.** It's 30 lines of changes
   and a 3-hour DGX run. The hyperparameter table in §3 has 5+ values
   off by 3-10×. There's a real chance Phase 1 alone gets us across the
   line. If it doesn't, the run still gives us a clean baseline against
   which to measure Phase 2.

2. **For the GiGPO anchor key — exact match or similarity match?**
   *My recommendation:* **exact match on the coarse 6-tuple in §4.3.**
   Similarity matching is for unstructured observations (web page
   contents, ALFWorld scene text). Ours is a clean grid + numeric stats;
   a 6-tuple is the right granularity.

3. **For Phase 3 SFT bootstrap — wait for Phase 2 result, or build
   the dataset in parallel?**
   *My recommendation:* **build the dataset in parallel.** It's a 1-hour
   task with no risk and the dataset is useful even if we don't end up
   doing SFT in Phase 3 (it's also useful as an offline benchmark).

4. **Should we abandon `.planning2/10_PPO_TRAJECTORY.md`?**
   *My recommendation:* **don't delete it, but mark it superseded.**
   It captures the PPO+critic option in case we need Phase 4. Add a
   header pointing here.

5. **Do you want me to attempt to use the `verl-agent` framework
   directly instead of porting GiGPO into our codebase?**
   *My recommendation:* **no.** verl-agent requires vLLM + FSDP + a
   specific env interface registration. Our V100 + Docker + LoRA stack
   is already working; adapting verl-agent to it is a side quest.
   Porting the 300 lines of GiGPO into our existing GRPOTrainer wrap is
   strictly less work.

---

## 8. Concrete next actions (in order)

1. **Mark plan 10 as superseded.** Add a 3-line header pointing here.
2. **Phase 1 commit**: hyperparameter recalibration in `training/train.py`.
   New defaults exactly as listed in §5/Phase 1. One commit, push.
3. **Run 5 on DGX**: ~3 hours wallclock. Eval the last 3 checkpoints.
4. **If Phase 1 fails** (mean_alive_at_end < 1.9): start Phase 2.
   - Port `gigpo.py`. Write the anchor key. Subclass GRPOTrainer.
   - Validation gates A+B on a 5-step toy run.
   - Run 6 on DGX.
5. **In parallel with Phase 1+2 (independent task):** build the SFT
   dataset (500 heuristic episodes → JSONL). 1 hour, no risk.

---

## 9. What I would tell you honestly

You've spent 80 hours on GRPO and gotten 0% survival, but you have
something the literature confirms: a clean pipeline that captures the
"Echo Trap" failure mode. That's not wasted time — it's *exactly* the
diagnostic loop you needed to credibly evaluate alternatives. The
literature now says the fix is either GiGPO or SFT-then-GRPO. Both fit
inside our existing infrastructure with bounded effort.

Phase 1 (hyperparameter fix) is **a 1-hour change with 5-10× corrections
across multiple knobs**. Even if you're skeptical of the rest of this
plan, Phase 1 alone is "almost free" relative to a DGX session.

If you want me to start, say so and I'll begin with Phase 1: modify the
training defaults, run the unit tests, push the commit. The DGX run is
yours to kick off.
