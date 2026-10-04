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
- [x] **W3 — Rebalance to v3.** Find the smallest knob set that hits: heuristic survival
  ~5-20%, oracle >= 60%, random ~0%. Write the table before/after into
  `.planning2/14_V3_BALANCE.md` with one line of reasoning per knob. Make v3 the default.
  Keep layout copies in `training/inference.py` and counts in `prompts.py` in sync.
  Ask the manager before changing anything about voting/infection roles (the social
  deduction part is the project's point).

  - Done 2026-10-04: default preset `v3-rc1` (rate 0.6, zombies move every 2nd step, chase radius 4,
    starting infected exempt). heuristic_v3 11%, oracle 83%, camp 73%, random 0%, heuristic_v2 0%.
    heuristic_v3 = rollout/baseline policy; prompts/rubric/postmortem follow cfg; bite events in prompt.
    Details: `.planning2/14_V3_BALANCE.md`.

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

**2026-10-04 status:** W3 done (d600c33): v3-rc1 default, heuristic_v3 11%, camp 73%, oracle 83%, random 0%.
**User chose the MINIMUM PATH to a first DGX run:** merge (done, manager) -> W6 -> GRPO run on v3-rc1
(survival only). W4, W5, W7, W8 come after the first run. Worker does ONLY W6, then stops.

### Phase B (W6 active now; W7/W8 queued)

- [ ] **W6 — Mid-episode start states.** Replace reset-only `build_scenario_dataset`:
  for each scenario pick seed N, roll a *behaviour mix* forward (planner / heuristic /
  epsilon-random per agent, chosen by a seeded rng) to time t, keep the state if A0 is alive.
  Choose t by **decision density** (see plan 13 research update): weight states where A0 has
  low water/food, a zombie within 3, a vote turn, or the radio/extraction window; cap routine
  "safe in safehouse, full stats" states at ~20%. Prompt = A0's real observation at t, with
  `[SEED:N][T:t][MIX:m]`. Reward fn replays deterministically to t (same rngs), applies the K
  model actions, continues with the **planner** (not the weak heuristic) for H steps (default
  H=25, flag), and scores A0's return over that window including any milestone/extraction
  reward that lands inside it. Must-have test: replay to t reproduces the exact state
  (positions, stats, zombies, food timers) for 50 random (N,t). Log the t histogram + state
  tags of the dataset to research_log. Training scenarios fix A0 as healthy (infected roles
  sampled from A1-A4 only, via a create_episode option); an infected A0 has a different
  objective and would muddy the survival gradient. Eval reports both A0-healthy and
  natural-role seeds.
- [ ] **W7 — SFT warm-start data + trainer.** `training/build_sft_dataset.py`: run the
  non-oracle planner over the W6 state distribution, write (prompt, completion) JSONL at A0
  turns, completion in the exact JSON format the RL prompt asks for (K-action array).
  `training/sft.py`: LoRA SFT with prompt tokens masked, same LoRA config as train.py, so the
  adapter loads via the existing warm-start path. Verify on CPU with a tiny model
  (e.g. `Qwen/Qwen2.5-0.5B-Instruct`, 20 steps) that loss drops and the output parses.
- [ ] **W8 — Closed-loop eval.** `training/eval.py` mode where the model drives A0 at every
  turn (re-plan every K turns), others run planner or heuristic (flag), plus a mode where the
  model drives *all* healthy agents. Report: extraction rate, A0 lifetime, healthy alive at
  end, survival, mean reward, action histogram; vs heuristic/planner/oracle on the SAME seeds.
  Writes to research_log and optionally records replays (W5) for the first 3 episodes.

## Manager queue
- [x] M1 — DGX infra. `~/zombiee-v3` + conda env `zombiee` (torch 2.5.1 cu121, trl 0.15.2, bnb 0.42.0 because glibc 2.17),
  Qwen2.5-3B cached, 84 tests pass, GPU smoke test OK (fp16, 6.5 GB peak, 1298-token prompt).
  Quota 90 GB: user approved deleting conda envs ec, eckv (Amazon ML challenge) + heever on 2026-10-04 -> ~13 GB free.
  Keep --save-total-limit <= 5 for training runs.
- [x] M2 — Reviewed W1 + W2 (reproduced independently); W3 merged into claude/phase-2-gigpo, 116 tests pass.
- [x] M3 — Spec Phase B tasks (mid-episode states, fixed horizon, SFT from planner).

## Done
- 2026-10-04 manager: ceiling probes. Env is unwinnable even for an oracle (0/100 reach t=100).
  Training prompts are all step-0 resets. See plan 13.
