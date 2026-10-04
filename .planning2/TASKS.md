# Task board — manager `zombiee-v2-ce`, worker `zombiee-v2-9f`

Rules
- Worker works on branch `v3-winnable-env` in the main checkout `D:\projs\extra\v2\zombiee-v2`
  (manager works in worktree `.claude/worktrees/wonderful-ritchie-21e41a`, branch `claude/phase-2-gigpo`).
- Small commits, authored as Sirjan, **no Co-Authored-By / AI trailers**. Push the branch after each task.
- Don't touch the unrelated staged files in the main checkout (RUN_ON_*.md, notebooks, .env.example, .gitignore).
- Run `python -m pytest -q` before every push. Don't run `docker build` anywhere.
- When a task is done: tick it here, add a one-line result, then message the manager.
- **Document everything** (user will build YouTube visualisations from it): every calibration/probe/eval
  run appends one JSON line to `research_log/experiments.jsonl` (date, git sha, balance config, policy,
  n_eps, all metrics, death causes) and a short entry to `research_log/LOG.md`. Raw per-episode results go
  in `research_log/data/`. Never overwrite old results.
- Plan + evidence: `.planning2/13_V3_MANAGER_PLAN.md`. Probe scripts: `tools/probes/`.

## Worker queue

- [x] **W1 — Balance config.** Move env balance constants (hunger/thirst thresholds and tick
  rates, infected hunger multiplier, zombie chase behaviour, wave schedule, MAX_ZOMBIES, food
  respawn delay, P_BITE, LATENT_DURATION, safehouse heal) into one `BalanceConfig` dataclass
  (e.g. `survivecity_v2_env/balance.py`). Defaults = today's values, so behaviour is identical.
  Prove it: `tools/probes/baseline.py` and `oracle.py` numbers unchanged; tests pass.
  - Done 2026-10-04 (142e45c, ccd8397): `survivecity_v2_env/balance.py` (`BalanceConfig`, preset `v2.2`,
    `get_balance`, fractional `meter_tick`). Env/game/spawn/infection/rubric read `state.balance`.
    Identity: sha256 of full heuristic/random/oracle trajectories (40 seeds each) identical before/after;
    baseline.py 16.02/12.98 and oracle.py 66.76/49.04 unchanged; golden-hash test pins v2.2. 97 tests pass.
- [x] **W2 — Calibration script.** `tools/calibrate.py`: runs random, current heuristic,
  camp planner, oracle for N=100 eps under a given BalanceConfig and prints one table
  (ep_len, A0 lifetime, healthy alive at end, survival, reached-t100, death causes).
  - Done 2026-10-04: `python tools/calibrate.py [--balance P] [--set k=json] [--policies ...] [--n] [--tag] [--note]`.
    Policies random/heuristic/camp/oracle, probe seed scheme (reproduces probe numbers exactly). Writes
    experiments.jsonl rows + data/<date>_calibrate_<tag>.json (per-episode) + LOG.md table; `--no-log`/`--no-md`.
    v2.2 baseline logged (tag v2.2-baseline): all 0% survival; oracle ep 66.8, A0 49.0. ~45 s for all 4 x 100.
- [ ] **W3 — Rebalance to v3.** Find the smallest knob set that hits: heuristic survival
  ~5-20%, oracle >= 60%, random ~0%. Write the table before/after into
  `.planning2/14_V3_BALANCE.md` with one line of reasoning per knob. Make v3 the default.
  Keep layout copies in `training/inference.py` and counts in `prompts.py` in sync.
  Ask the manager before changing anything about voting/infection roles (the social
  deduction part is the project's point).

  - PAUSED 2026-10-04 at a clean point (usage pause). Done: knobs `zombie_move_every`,
    `starting_infected_progression` (approved), hp bound follows hp_max, `tools/sweep_balance.py`,
    preset `v3-rc1` (oracle 83%, camp 73%, random 0%, heuristic 0%), logged sweeps + findings.
    Blocked on a decision: the current heuristic is 0% in every balance (3 bugs + no zombie
    avoidance; see LOG.md). A fixed + avoiding prototype gets 13% at v3-rc1.
    Not done: make v3 the default; sync prompts.py / rubric.py thresholds / postmortem.py latent;
    bite events are only in metadata.bite_history, not in prompt text.
- [ ] **W4 — Radio + extraction objective (plan 13, Phase A2; replaces the rescue-at-90 idea).**
  Radio at t=60 names an extraction corner (seeded); helicopter t=85-90 extracts healthy
  agents in the zone; any infected agent in the zone = extraction fails (infected win);
  small milestone bonus at t=30/60. Weights in config. Prompt shows radio info + turns left.
  Extend the oracle planner to do the extraction run so calibration can measure it.
  Do after W3 has a winnable survival baseline. Propose reward weights to the manager first.
- [ ] **W5 — Replay recorder (for YouTube visuals).** `tools/record_episode.py`: runs one
  episode under a given policy + BalanceConfig and writes `research_log/replays/<name>.json`
  with every step: grid, agents (pos, hp, hunger, thirst, infection, inventory, action),
  zombies, events (bites, deaths, votes, radio, extraction). Record heuristic vs oracle on the
  same seeds, old balance vs v3.

(Phase B tasks get added after W3/W4 land.)

## Manager queue
- [ ] M1 — DGX infra: fresh clone, conda env, CUDA check, Qwen2.5-3B download.
- [ ] M2 — Review W1-W3 diffs; re-run calibration independently.
- [ ] M3 — Spec Phase B tasks (mid-episode states, fixed horizon, SFT from planner).

## Done
- 2026-10-04 manager: ceiling probes. Env is unwinnable even for an oracle (0/100 reach t=100).
  Training prompts are all step-0 resets. See plan 13.
