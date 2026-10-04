# Task board — manager `zombiee-v2-ce`, worker `zombiee-v2-9f`

Rules
- Worker works on branch `v3-winnable-env` in the main checkout `D:\projs\extra\v2\zombiee-v2`
  (manager works in worktree `.claude/worktrees/wonderful-ritchie-21e41a`, branch `claude/phase-2-gigpo`).
- Small commits, authored as Sirjan, **no Co-Authored-By / AI trailers**. Push the branch after each task.
- Don't touch the unrelated staged files in the main checkout (RUN_ON_*.md, notebooks, .env.example, .gitignore).
- Run `python -m pytest -q` before every push. Don't run `docker build` anywhere.
- When a task is done: tick it here, add a one-line result, then message the manager.
- Plan + evidence: `.planning2/13_V3_MANAGER_PLAN.md`. Probe scripts: `tools/probes/`.

## Worker queue

- [ ] **W1 — Balance config.** Move env balance constants (hunger/thirst thresholds and tick
  rates, infected hunger multiplier, zombie chase behaviour, wave schedule, MAX_ZOMBIES, food
  respawn delay, P_BITE, LATENT_DURATION, safehouse heal) into one `BalanceConfig` dataclass
  (e.g. `survivecity_v2_env/balance.py`). Defaults = today's values, so behaviour is identical.
  Prove it: `tools/probes/baseline.py` and `oracle.py` numbers unchanged; tests pass.
- [ ] **W2 — Calibration script.** `tools/calibrate.py`: runs random, current heuristic,
  camp planner, oracle for N=100 eps under a given BalanceConfig and prints one table
  (ep_len, A0 lifetime, healthy alive at end, survival, reached-t100, death causes).
- [ ] **W3 — Rebalance to v3.** Find the smallest knob set that hits: heuristic survival
  ~5-20%, oracle >= 60%, random ~0%. Write the table before/after into
  `.planning2/14_V3_BALANCE.md` with one line of reasoning per knob. Make v3 the default.
  Keep layout copies in `training/inference.py` and counts in `prompts.py` in sync.
  Ask the manager before changing anything about voting/infection roles (the social
  deduction part is the project's point).

- [ ] **W4 — Rescue objective (plan 13, Phase A2).** Episode ends at t=90 with "rescue":
  healthy agents alive and inside the safehouse at t=90 are rescued (big terminal reward),
  milestone bonuses at t=30/60, infected win if they outnumber healthy at rescue. Put the
  weights in BalanceConfig/reward config. Show the prompt text includes turns-to-rescue.
  Recalibrate (W2 script) with rescue rate as the headline. Can be done together with W3,
  since rebalancing should target the rescue rate. Propose weights to the manager before
  removing any existing rubric.

(Phase B tasks get added after W3/W4 land.)

## Manager queue
- [ ] M1 — DGX infra: fresh clone, conda env, CUDA check, Qwen2.5-3B download.
- [ ] M2 — Review W1-W3 diffs; re-run calibration independently.
- [ ] M3 — Spec Phase B tasks (mid-episode states, fixed horizon, SFT from planner).

## Done
- 2026-10-04 manager: ceiling probes. Env is unwinnable even for an oracle (0/100 reach t=100).
  Training prompts are all step-0 resets. See plan 13.
