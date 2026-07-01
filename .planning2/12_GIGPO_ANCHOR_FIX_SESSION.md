# 12 — Session notes: GiGPO anchor fix (real step-level credit assignment)

**Status:** implemented + committed + pushed on `claude/phase-2-gigpo`; UNVALIDATED on DGX
**Date:** 2026-06-29 → 2026-06-30
**Follows:** `11_RESEARCH_FINDINGS_AND_REVISED_PLAN.md` (Phase 2 = GiGPO)

---

## TL;DR

The Phase-2 GiGPO code (commits `dcc1739`, `3db47b5`, `a86f5b4`) ran and passed
its unit tests, but was **algorithmically inert** — it degenerated to GRPO. We
diagnosed this on CPU (no DGX spend), fixed it, and re-validated that the GiGPO
mechanism now actually fires. The Phase-0 diagnostic gate did its job.

## Algorithm decision (re-confirmed with fresh 2026 literature)

- **PPO + critic**: rejected — worst results in every recent table (54% vs GRPO
  70% on ALFWorld-1.5B) and the most code. Stays as a last-resort only.
- **GiGPO** (NeurIPS'25, arxiv 2505.10978): chosen — critic-free, +12–18pp over
  GRPO, the only method validated at exactly Qwen2.5-3B, released reference code.
- **GAGPO** (May'26, 2605.13217): a strict advantage-step upgrade of GiGPO
  (+5–7pp), but no code / no 3B validation → deferred future delta, not now.
- **Scope: single-agent (A0 only)**, per user decision. Multi-agent (C3/CCPO)
  deferred — no 3B validation, no code, needs a new eval harness.

## The bug: GiGPO was inert

`train.py` reward fn snapshotted the anchor only at `env.reset()` — a **constant**
state across a same-seed GRPO group. Consequences, proven with a CPU probe on the
real env:
- **1** distinct anchor key across 16 seeds (agent always spawns identically).
- Every GRPO group → **one cluster of size = num_generations (8)**.
- Phase-0 GO gate (mean cluster size ∈ [1.5, 6]) **FAILED at 8.0**.
- The step advantage was just a redundant copy of the first-step reward already
  inside the episode advantage. GiGPO's cross-time state matching was absent.
- **Doubly inert:** `--prefix-actions` defaults to 1 → only one model step exists.

## The fix

1. **Per-step anchor capture** (`train.py`): snapshot the anchor at each model
   action's pre-state; side-channel is now a list-of-`(anchor, step_reward)` per
   generation. Error/empty paths append `[]`.
2. **`compose_gigpo_advantages_multistep`** (`gigpo.py`): pool all (gen, step)
   entries, cluster by (prompt, anchor) → same state at different turns across
   trajectories forms one step-level group → reduce to one scalar/generation.
3. **Refined `anchor_key_for_agent0`**: exact (row,col) + hunger//2 + thirst//2,
   **no step bucket** (so cross-time matching works; hunger/thirst still separate
   genuinely different situations).
4. **Trainer** (`gigpo_trainer.py`): `_prepare_inputs` consumes the per-step lists.
5. **Guards/docs**: warn on `gigpo + prefix_actions=1`; `--gigpo-zscore` help
   records the scale finding.

## Validation (all CPU, no DGX)

- **84/84 tests pass** (added 6 multistep cases).
- Revised anchor probe: **61 distinct keys**, **mean cluster size 2.33** (paper
  ~2.4), `step_adv_std > 0` → **GO gate PASSES**.
- **Producer→consumer contract validated end-to-end**: real `create_reward_fn`
  populates the side-channel correctly (n entries, K/gen), anchors evolve per
  step, compose yields real non-zero signal.
- **Scale finding:** env per-step `raw_reward` is tiny → mean_norm step advantage
  ~0.001 (negligible vs GRPO's unit-scale episode advantage); z-score ~0.83.
  ⇒ **`--gigpo-zscore` is required** on this env.

## Commits (pushed to `origin/claude/phase-2-gigpo`)

- `08a20a5` gigpo: per-step anchors + multistep compose + refined key + tests
- `5c8b8bf` train: wire per-step capture + consume multistep advantage + guards

## Next step (needs DGX)

**Run 6:** `python -m training.train --adv-estimator gigpo --prefix-actions 5 --gigpo-zscore`
Watch `[gigpo]` log line: `mean_size` ~2–3, `step_adv_std > 0`, and `kl > 0` by
step 5. Eval with the standard `training.eval`; compare `mean_alive_at_end`
against the GRPO Phase-1 baseline and the pure-heuristic ceiling.

**Honest caveat:** this fixes the credit-assignment lever only. If the binding
constraint is the env/starvation ceiling (pure-heuristic baseline dies ~step 16),
a sharper gradient won't overcome it. Run 6 tells us which constraint binds.
