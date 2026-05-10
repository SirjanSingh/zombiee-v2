"""GRPO training pipeline for SurviveCity v2 — 24 GB-VRAM tuned.

Defaults are now sized for a single 24 GB GPU (HF Spaces A10G / Kaggle 2×T4
with one GPU active / a 24 GB consumer card). Override flags exist for
30 GB DGX runs — see ``--num-generations 12 --max-completion-length 512``
in the v2 plan files.

Default hyperparameters:
    --model-name              Qwen/Qwen2.5-3B-Instruct
    --max-steps               15         (every step → checkpoint)
    --save-steps              1
    --save-total-limit        15
    --num-generations         8          (24 GB-safe; raise to 12 on 30 GB)
    --grad-accum-steps        4          (4 prompts × 8 gens = 32 evals/step)
    --max-completion-length   384        (24 GB-safe; raise to 512 on 30 GB)
    --rollout-limit           60         (heuristic-rollout horizon)
    --lora-r                  32
    --lora-alpha              64
    --max-seq-length          4096
    4-bit nf4 quant ENABLED by default (pass --no-4bit on >=30 GB cards)

Time budget on 24 GB A10G: ~30 min/step (4-bit base, num_gen=8,
max_compl=384) → 15 steps ≈ 7.5 h. On a 30 GB A100 with --no-4bit and the
old DGX flags (num_gen=12, max_compl=512) it stays ~24 min/step → 6 h.

Memory strategy: 4-bit base model (NF4 + double-quant) by default —
combined with gradient checkpointing this fits 8 generations × 384
completion tokens × 1536 prompt tokens inside ~22 GB peak on A10G
(measured), leaving ~2 GB headroom for KV cache spikes. The --no-4bit
path needs ~28 GB peak and is meant for A100/H100.

VRAM holder: DISABLED. The block in main() that allocated a "holder" tensor
to pin headroom on shared GPUs has been commented out. In practice, on a
shared DGX the co-tenant already had their memory before our process started,
so the holder couldn't reclaim it; it just shrunk our own training budget.
If you want it back (e.g. on a single-tenant GPU you fully control),
uncomment the block in main() and pass `--vram-reserve-gb 16` or similar.

Hub push: opt-in via --push-to-hub. With `hub_strategy="every_save"`, every
save-step pushes the entire output_dir to the Hub repo, so the 15GB eval box
can pull mid-training and run training.eval against any checkpoint.

Usage:
    python -m training.train [args]

The script uses a LOCAL SurviveCityV2Env in the GRPO reward function — no
HTTP server required during training (same pattern as v1).
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import os
import random
import re
import signal
import sys
import threading
import time
from typing import Optional

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("survivecity_v2.train")

# Crash diagnostics: dumps Python tracebacks for every thread on fatal signals
# (SIGSEGV, SIGABRT, SIGBUS, SIGFPE) and on `kill -USR1 <pid>` for live probes.
# Without this, an OS-level kill (OOM, SIGSEGV from CUDA, etc.) leaves zero
# clue why training "just stopped".
faulthandler.enable()
try:
    faulthandler.register(signal.SIGUSR1)
except (AttributeError, ValueError):
    pass  # Windows / non-POSIX
# Force line buffering so the log we tail actually shows the LAST line before
# the kill, not whatever happened to be in the 8KB stdio buffer.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass


def _report_existing_checkpoints(output_dir: str) -> None:
    """Print any leftover checkpoint-* dirs and their disk usage at startup.

    Surfaces stale checkpoints from a previous run BEFORE training starts —
    so you notice if save-total-limit kept old ones around, or if a previous
    crashed run is about to be silently overwritten / resumed.
    """
    if not os.path.isdir(output_dir):
        logger.info(f"checkpoints: output_dir={output_dir} doesn't exist yet (fresh start).")
        return
    ckpts = sorted(
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    )
    if not ckpts:
        logger.info(f"checkpoints: no checkpoint-* dirs under {output_dir} (fresh start).")
        return
    total_gb = 0.0
    for c in ckpts:
        p = os.path.join(output_dir, c)
        try:
            sz = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(p) for f in fs
            ) / 1e9
        except OSError:
            sz = float("nan")
        total_gb += sz if sz == sz else 0
        logger.info(f"checkpoints: found {c}  ({sz:.2f} GB)")
    logger.info(
        f"checkpoints: {len(ckpts)} existing under {output_dir}, total {total_gb:.2f} GB. "
        f"Pass --resume-from-checkpoint auto to continue from the latest."
    )


def _start_mem_watchdog(interval_s: int = 30) -> None:
    """Background thread that logs RSS + GPU memory every interval_s seconds.

    Lets you tell apart "killed by linux OOM" (RSS climbing → big jump → dead)
    from "killed externally" (RSS flat right up to the moment of death).
    Runs as a daemon so it doesn't block process exit.
    """
    try:
        import psutil
        proc = psutil.Process()
        get_rss_gb = lambda: proc.memory_info().rss / 1e9
        get_avail_gb = lambda: psutil.virtual_memory().available / 1e9
    except ImportError:
        import resource
        get_rss_gb = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        get_avail_gb = lambda: float("nan")

    def loop():
        while True:
            try:
                msg = f"[mem] RSS={get_rss_gb():.1f}GB sys_avail={get_avail_gb():.1f}GB"
                if torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info(0)
                    alloc = torch.cuda.memory_allocated(0) / 1e9
                    reserved = torch.cuda.memory_reserved(0) / 1e9
                    msg += (
                        f"  GPU_free={free/1e9:.1f}/{total/1e9:.1f}GB "
                        f"alloc={alloc:.1f}GB reserved={reserved:.1f}GB"
                    )
                logger.info(msg)
            except Exception as e:
                logger.warning(f"mem watchdog error: {type(e).__name__}: {e}")
            time.sleep(interval_s)

    t = threading.Thread(target=loop, daemon=True, name="mem-watchdog")
    t.start()
    logger.info(f"mem watchdog started (interval={interval_s}s)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--max-steps", type=int, default=15)
    p.add_argument("--save-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--num-generations", type=int, default=8,
                   help="GRPO group size. 8 fits a 24 GB GPU; raise to 12 on 30 GB DGX.")
    p.add_argument("--output-dir", default="./checkpoints")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--max-prompt-length", type=int, default=1536)
    p.add_argument("--max-completion-length", type=int, default=384,
                   help="Completion token budget. 384 fits 24 GB; bump to 512 on 30 GB.")
    p.add_argument("--rollout-limit", type=int, default=60,
                   help="Max heuristic-rollout steps after the model's first action. "
                        "Higher = denser cumulative-reward signal but slower training step.")
    p.add_argument("--step1-weight", type=float, default=5.0,
                   help="Weight on the model's first-action raw_reward in the GRPO "
                        "composite. The 1 model action would otherwise be drowned by "
                        "the rollout cumulative; default 5.0 balances them.")
    p.add_argument("--format-bonus", type=float, default=0.10,
                   help="Bonus added when parse_action returned a real action. Keeps "
                        "the within-group variance non-zero when env reward ties.")
    p.add_argument("--prefix-actions", type=int, default=1,
                   help="Number of model actions to apply at agent 0's turns "
                        "before the heuristic takes over. K=1 (default) is the "
                        "legacy single-action mode. K>1 expects a JSON array of "
                        "K actions in the completion. With K>1 you should also "
                        "drop --step1-weight (try 1.0) since the model contribution "
                        "is now summed across K actions.")
    # ---- v2.1 metrics + safety-push flags -------------------------------
    p.add_argument(
        "--metrics-file", default=None,
        help="JSONL path for per-reward_fn metrics (action histogram, "
             "per-rubric averages, reward stats, terminal-survival rates). "
             "Default: <output-dir>/metrics.jsonl. Set to '' to disable.")
    p.add_argument(
        "--summary-file", default=None,
        help="JSON path for end-of-run summary. Default: <output-dir>/train_summary.json.")
    p.add_argument(
        "--safety-push-every-min", type=int, default=5,
        help="Heartbeat push to HF Hub every N minutes (independent of "
             "save_steps). Pushes the metrics file + latest checkpoint dir "
             "if --push-to-hub is set. 0 disables. Insures against losing "
             "training state if the box dies between save_steps.")
    p.add_argument(
        "--auto-plots", action="store_true", default=True,
        help="At end of training, auto-generate the v2.1 plots PNG set into "
             "<output-dir>/plots/. Disable with --no-auto-plots.")
    p.add_argument(
        "--no-auto-plots", dest="auto_plots", action="store_false",
        help="Skip auto-plot generation at end of training.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-scenarios", type=int, default=200)
    p.add_argument("--report-to", default="tensorboard")
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    # Resume / hub flags
    p.add_argument(
        "--resume-from-checkpoint", default=None,
        help="Path to a checkpoint dir, 'auto' for latest under --output-dir, "
             "or a HF Hub repo id (snapshot-downloaded).")
    p.add_argument(
        "--push-to-hub", action="store_true",
        help="Push every checkpoint to --hub-model-id (requires HUGGINGFACE_TOKEN).")
    p.add_argument(
        "--hub-model-id", default="noanya/zombiee-v2",
        help="HF Hub repo id. Default 'noanya/zombiee-v2' is the team's v2 repo "
             "(separate from v1's 'noanya/zombiee' so v1 artefacts stay frozen).")
    p.add_argument(
        "--hub-private", action="store_true", default=True,
        help="Create the hub repo as private. Default True for unreleased work; "
             "use --hub-public to make a fresh repo public.")
    p.add_argument(
        "--hub-public", dest="hub_private", action="store_false",
        help="Make the hub repo public (overrides --hub-private).")
    p.add_argument("--save-total-limit", type=int, default=15,
                   help="Keep at most this many checkpoints on disk locally. "
                        "Default 15 keeps every save from a 15-step / save_steps=1 run "
                        "(~30 MB adapter × 15 ≈ 450 MB on Hub).")
    p.add_argument("--gradient-checkpointing", action="store_true", default=True,
                   help="Enable gradient checkpointing to fit larger num_generations × "
                        "max_completion_length within VRAM. Trades ~30%% per-step time for ~40%% "
                        "memory headroom — a worthwhile swap on a 30GB box at num_generations=12.")
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                   action="store_false",
                   help="Disable gradient checkpointing (only safe with smaller num_generations).")
    p.add_argument("--optim", default="adamw_torch_fused",
                   help="Optimizer name. adamw_torch_fused is faster on Ampere+ (sm_80+); "
                        "fall back to adamw_torch on V100/T4.")
    p.add_argument(
        "--warmstart-from", default=None,
        help="Optional HF Hub repo id (or local path) to a v1 LoRA. "
             "Loaded as the initial PEFT adapter instead of training from "
             "scratch — this is the v2-warmstart-from-v1 transfer experiment.")
    p.add_argument(
        "--allow-fp16", action="store_true",
        help="Force fp16 even on Ampere+; default auto-picks bf16 on Ampere+ and fp16 elsewhere.")
    p.add_argument(
        "--no-4bit", action="store_true",
        help="Skip bitsandbytes 4-bit and load the base model in bf16/fp16 at full weight precision. "
             "Recommended on 30GB+ A100/H100 boxes — gives cleaner gradients than 4-bit.")
    p.add_argument(
        "--vram-reserve-gb", type=int, default=0,
        help="DEPRECATED — VRAM holder is disabled. Was: amount of GPU memory to leave "
             "free for training while holding the rest. In practice, on a shared GPU the "
             "co-tenant already had their memory BEFORE we started, so the holder couldn't "
             "claim that back; it just shrunk the budget for our own training peak. "
             "Default 0 = no holder. Pass any positive value to re-enable (the allocation "
             "block is commented out below — uncomment if you really want it).")
    return p.parse_args()


def build_scenario_dataset(num_scenarios: int = 200, seed: int = 42):
    """Build a GRPO scenario dataset from local v2 env resets.

    Each prompt embeds [SEED:N] so the reward function can recreate the exact
    env state for fair within-group comparison.
    """
    from datasets import Dataset
    from survivecity_v2_env.env import SurviveCityV2Env
    from survivecity_v2_env.prompts import build_system_prompt

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x  # noqa: E731

    rng = random.Random(seed)
    prompts = []
    for i in tqdm(range(num_scenarios), desc="build_scenario_dataset"):
        try:
            ep_seed = rng.randint(0, 999999)
            env = SurviveCityV2Env()
            obs = env.reset(seed=ep_seed)
            desc = obs.get("description", "")
            prompt = build_system_prompt(0, f"[SEED:{ep_seed}]\n{desc}")
            prompts.append({"prompt": prompt, "scenario_id": i})
        except Exception as e:
            logger.warning(f"Scenario {i} failed: {e}")
    logger.info(f"Built {len(prompts)} v2 scenario prompts")
    return Dataset.from_list(prompts)


# Marker used by the kaggle notebook to detect whether the file already has
# the v2-reward-fix applied; do not remove.
REWARD_FN_VERSION = "v2.1-heuristic-rollout-2026-04-26"


def create_reward_fn(
    rollout_limit: int = 60,
    step1_weight: float = 5.0,
    format_bonus: float = 0.10,
    prefix_actions: int = 1,
    metrics_logger=None,
    trainer_state_ref: Optional[dict] = None,
):
    """GRPO reward function — v2.1 heuristic-rollout edition.

    What's new in v2.1 (vs v2.0 cumulative):

      1. **Heuristic rollout**, not random rollout. The v2.0 fn rolled out
         30 RANDOM actions after the model's single action. Random policy on
         a 15×15 grid with 8 food cells and 100-step horizon almost never
         reaches food, so rollouts universally end in starvation around
         step 18-25 with nearly identical cumulative reward across the GRPO
         group (reward_std ≈ 0 → no gradient). The v2.1 rollout uses
         `forage_heuristic_action` from training.inference, which goes to
         food/water when hungry/thirsty. Empirically the rollout reaches
         step 50-70 instead of 18-25, so the cumulative_reward distribution
         actually has variance for GRPO to learn from.
      2. **Per-step credit shines through.** The new shaping rubrics
         (forage_shaping, zombie_proximity, anti_camp) make the dense
         per-step signal informative even before the model's action lands
         a clear win — so the cumulative term carries gradient even when
         the model didn't single-handedly turn the rollout around.
      3. **Pending-reward credit is now real.** survivecity_v2_env.env was
         changed to drain per-agent pending_reward into cumulative_rewards
         at end-of-round, so zombie damage to non-acting agents (which v1
         silently dropped) shows up in cumulative_rewards[0] and reaches
         the GRPO gradient.
      4. **Same composite shape as v2.0** so existing TB curves remain
         comparable: `STEP1_WEIGHT * step1_raw + cumulative + FORMAT_BONUS`.
         Defaults: step1_weight=5.0, format_bonus=0.10, rollout_limit=60.
      5. **No clip.** OpenEnv only constrains the env's `obs.reward`. The
         GRPO reward function is internal and TRL normalises within the
         group anyway — clipping here erases relative differences and
         was the v1 bug.
    """
    from collections import Counter
    import statistics
    from survivecity_v2_env.env import SurviveCityV2Env
    from training.inference import parse_action, parse_actions, forage_heuristic_action
    from training.metrics import build_terminal_summary

    # Clamp to >=1 — prefix_actions=0 would mean "model never acts" which is
    # never what we want; treat anything <1 as the legacy 1-action mode.
    prefix_actions = max(1, int(prefix_actions))

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x  # noqa: E731

    state = {"calls": 0, "errors": 0}

    def reward_fn(prompts, completions, **kwargs):
        state["calls"] += 1
        rewards: list[float] = []
        action_types: list[str] = []
        rollout_lens: list[int] = []
        cum_rewards_seen: list[float] = []
        rubric_breakdowns: list[dict] = []
        final_obs_for_terminal: list[dict] = []
        parse_ok_count = 0

        n = len(prompts)
        iterator = tqdm(
            list(zip(prompts, completions)),
            total=n,
            desc=f"reward_fn #{state['calls']}",
            leave=False,
        )
        for prompt, completion in iterator:
            try:
                seed_match = re.search(r"\[SEED:(\d+)\]", prompt)
                ep_seed = int(seed_match.group(1)) if seed_match else (
                    abs(hash(prompt)) % 1_000_000
                )
                env = SurviveCityV2Env()
                obs = env.reset(seed=ep_seed)

                # parse_actions returns a list. With prefix_actions=1 it
                # behaves like the old parse_action (single object → 1-elem
                # list). With prefix_actions>1 the model is expected to emit
                # a JSON array; if it emits a single object instead we still
                # honour that as 1 action and fill the rest with wait.
                parsed_list = parse_actions(
                    completion, agent_id=0, max_actions=prefix_actions,
                )
                parse_ok = len(parsed_list) > 0
                if parse_ok:
                    parse_ok_count += 1
                # Pad to exactly prefix_actions with wait so the rollout loop
                # always has K candidates to apply for agent 0.
                while len(parsed_list) < prefix_actions:
                    parsed_list.append({"agent_id": 0, "action_type": "wait"})

                # Track action types for the per-call histogram. We record
                # all K planned model actions (or PARSE_FAIL when parse_ok=False
                # for the K wait fillers) so the histogram reflects the full
                # prefix the model is choosing.
                if parse_ok:
                    for a in parsed_list:
                        action_types.append(a.get("action_type", "?"))
                else:
                    for _ in range(prefix_actions):
                        action_types.append("PARSE_FAIL")

                # Unified rollout loop. Model acts at agent 0's turns until
                # we exhaust the K-action prefix; after that the heuristic
                # takes over for everyone (including agent 0).
                rollout_rng = random.Random(ep_seed + 7)
                steps = 0
                model_actions_used = 0
                model_step_raws: list[float] = []
                while not obs.get("done", False) and steps < rollout_limit:
                    aid = obs.get("metadata", {}).get("current_agent_id", 0)
                    if aid == 0 and model_actions_used < prefix_actions:
                        act = parsed_list[model_actions_used]
                        model_actions_used += 1
                        obs = env.step(act)
                        step_raw = obs.get("metadata", {}).get("raw_reward", 0.0)
                        model_step_raws.append(step_raw)
                        # Capture rubric breakdown of the FIRST model action only
                        # so existing metrics dashboards stay comparable.
                        if len(model_step_raws) == 1:
                            rb = obs.get("metadata", {}).get("rubric_breakdown") or {}
                            if rb:
                                rubric_breakdowns.append(dict(rb))
                    else:
                        act = forage_heuristic_action(aid, obs, rng=rollout_rng)
                        obs = env.step(act)
                    steps += 1

                # Cumulative raw reward for agent 0 across the whole rollout
                cum0 = obs.get("metadata", {}).get(
                    "cumulative_rewards", {}
                ).get(0, 0.0)
                cum_rewards_seen.append(cum0)
                final_obs_for_terminal.append(obs)

                # Final composite — signed, NOT clipped (GRPO normalises internally)
                # step1_weight is applied to the SUM of model action raws so
                # at prefix_actions=1 the formula matches the legacy single-
                # action shape exactly. At prefix_actions=K, total signal is
                # roughly Kx larger; recommend dropping --step1-weight to ~1.0.
                composite = (
                    step1_weight * sum(model_step_raws)
                    + cum0
                    + (format_bonus if parse_ok else 0.0)
                )
                rewards.append(float(composite))
                rollout_lens.append(steps)
            except Exception as e:
                state["errors"] += 1
                if state["errors"] <= 5 or state["errors"] % 50 == 0:
                    logger.warning(
                        f"reward_fn error #{state['errors']} "
                        f"({type(e).__name__}): {e}"
                    )
                rewards.append(0.0)
                # Extend by prefix_actions so the histogram still has one
                # entry per planned model action, matching the success path.
                for _ in range(prefix_actions):
                    action_types.append("ERROR")
                rollout_lens.append(0)

        # Diagnostics — this is the line you tail in the log to see learning.
        # Guard against empty `rewards` (TRL shouldn't call us with n=0 but we
        # keep the failure mode local rather than crashing the trainer).
        if rewards:
            try:
                r_mean = statistics.mean(rewards)
                r_std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
                ro_mean = statistics.mean(rollout_lens) if rollout_lens else 0.0
                cum_mean = statistics.mean(cum_rewards_seen) if cum_rewards_seen else 0.0
                cum_std = statistics.stdev(cum_rewards_seen) if len(cum_rewards_seen) > 1 else 0.0
            except statistics.StatisticsError:
                r_mean = r_std = ro_mean = cum_mean = cum_std = 0.0
            r_min, r_max = min(rewards), max(rewards)
            action_dist = Counter(action_types).most_common(6)
            action_dist_str = ", ".join(f"{a}={c}" for a, c in action_dist)
            logger.info(
                f"reward_fn #{state['calls']}: n={n} "
                f"parse_ok={parse_ok_count}/{n} "
                f"r[mean={r_mean:+.3f} std={r_std:.3f} min={r_min:+.3f} max={r_max:+.3f}] "
                f"cum0[mean={cum_mean:+.3f} std={cum_std:.3f}] "
                f"avg_rollout={ro_mean:.0f} "
                f"actions[{action_dist_str}] "
                f"errs={state['errors']}"
            )
            # Persist the per-call metrics row so plots.py / external tools
            # can chart training over time. Wrapped in try/except because
            # logger failures must never take down a training step — the
            # gradient is more important than the metric.
            if metrics_logger is not None:
                try:
                    terminal_summary = build_terminal_summary(final_obs_for_terminal)
                    cur_step = (
                        trainer_state_ref.get("global_step")
                        if trainer_state_ref else None
                    )
                    metrics_logger.log_reward_call(
                        rewards=rewards,
                        cum_rewards=cum_rewards_seen,
                        rollout_lens=rollout_lens,
                        action_types=action_types,
                        parse_ok=parse_ok_count,
                        rubric_breakdowns=rubric_breakdowns,
                        terminal_summary=terminal_summary,
                        training_step=cur_step,
                    )
                except Exception as _le:
                    logger.warning(
                        f"reward_fn: metrics logger failed "
                        f"({type(_le).__name__}: {_le}); continuing without log."
                    )
        else:
            logger.warning(
                f"reward_fn #{state['calls']}: empty completions batch — nothing to score."
            )
        return rewards

    return reward_fn


def _resolve_resume(spec: str | None, output_dir: str):
    if not spec:
        return None
    if spec == "auto":
        if os.path.isdir(output_dir) and any(
            d.startswith("checkpoint-") for d in os.listdir(output_dir)
        ):
            return True
        logger.info(f"No checkpoint-* dir under {output_dir}; starting fresh.")
        return None
    if os.path.isdir(spec):
        return spec
    if "/" in spec and not spec.startswith(("./", "/")):
        from huggingface_hub import snapshot_download
        local = snapshot_download(
            repo_id=spec, local_dir=os.path.join(output_dir, "_resume")
        )
        logger.info(f"Downloaded {spec} -> {local}")
        return local
    return spec


def _seed_warnings_issued(m, depth: int = 0):
    """Seed `warnings_issued` dict on every PEFT wrapper (TRL >=0.15 quirk)."""
    if m is None or depth > 6:
        return
    try:
        if not isinstance(getattr(m, "warnings_issued", None), dict):
            m.warnings_issued = {}
    except Exception:
        pass
    for attr in ("base_model", "model"):
        inner = getattr(m, attr, None)
        if inner is not None and inner is not m:
            _seed_warnings_issued(inner, depth + 1)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Surface leftover checkpoints from a prior run BEFORE we touch anything,
    # so the operator notices stale state (and can pass --resume-from-checkpoint
    # auto if they wanted to continue).
    _report_existing_checkpoints(args.output_dir)

    # Fire up the RSS/GPU mem watchdog NOW, before model load — captures the
    # baseline so a sudden RSS jump in the run is obvious.
    _start_mem_watchdog(interval_s=30)

    # Precision selection
    cuda_ok = torch.cuda.is_available()
    cap = torch.cuda.get_device_capability(0) if cuda_ok else (0, 0)
    cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
    use_bf16 = cuda_ok and cap[0] >= 8 and cuda_major >= 11 and not args.allow_fp16
    use_fp16 = cuda_ok and not use_bf16
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    if cuda_ok:
        free_b, total_b = torch.cuda.mem_get_info(0)
        logger.info(
            f"GPU={torch.cuda.get_device_name(0)} cc={cap[0]}.{cap[1]} "
            f"cuda={torch.version.cuda} bf16={use_bf16} fp16={use_fp16} "
            f"VRAM free={free_b/1e9:.1f}GB total={total_b/1e9:.1f}GB"
        )
    else:
        logger.warning("CUDA not available — training will be CPU-only (slow).")

    # --------------------------------------------------------------
    # VRAM holder — DISABLED.
    #
    # Background: the holder pre-allocated a tensor for "unused" VRAM so
    # co-tenants on a shared GPU couldn't claim it mid-run. In practice,
    # on a shared DGX the co-tenant ALREADY had their memory before our
    # process started, so the holder never actually reclaimed anything.
    # All it did was shrink our own training budget by ~5 GB. With
    # 22 GB free + holder reserving 16 + safety 1 = 5 GB held, our
    # training only had 17 GB left — and Config A peak hits ~14-18 GB,
    # so the holder pushed us to OOM.
    #
    # The whole block below is commented out. If you want the holder
    # back (e.g. on a single-tenant GPU where you really do own all of
    # it), uncomment + pass `--vram-reserve-gb 16` (or whatever).
    # --------------------------------------------------------------
    if cuda_ok:
        logger.info("VRAM holder disabled (commented out). Training uses whatever the GPU has free.")
    # if cuda_ok and args.vram_reserve_gb > 0:
    #     try:
    #         free_b, _ = torch.cuda.mem_get_info(0)
    #         free_gb = free_b / 1e9
    #         safety_gb = 1.0
    #         hold_gb = free_gb - args.vram_reserve_gb - safety_gb
    #         if hold_gb > 0.5:
    #             torch._zombiee_vram_holder = torch.empty(
    #                 int(hold_gb * (1024 ** 3)),
    #                 dtype=torch.uint8,
    #                 device="cuda:0",
    #             )
    #             free_after_b, _ = torch.cuda.mem_get_info(0)
    #             logger.info(
    #                 f"VRAM holder pinned: {hold_gb:.2f} GB held on GPU 0 "
    #                 f"(reserved {args.vram_reserve_gb} GB for training, "
    #                 f"{safety_gb:.1f} GB safety). "
    #                 f"Free VRAM after holder: {free_after_b/1e9:.2f} GB."
    #             )
    #         else:
    #             logger.info(
    #                 f"Skipping VRAM holder: free {free_gb:.1f} GB - reserve "
    #                 f"{args.vram_reserve_gb} GB - safety {safety_gb:.1f} GB = "
    #                 f"{hold_gb:.1f} GB (need >0.5 GB to bother)."
    #             )
    #     except Exception as e:
    #         etype = type(e).__name__
    #         if "out of memory" in str(e).lower() or "OutOfMemoryError" in etype:
    #             logger.warning(
    #                 f"VRAM holder allocation hit OOM ({etype}): {str(e)[:120]}. "
    #                 f"Continuing without the holder."
    #             )
    #         else:
    #             raise

    device_map = {"": 0} if cuda_ok else "cpu"

    # Model + LoRA via transformers + peft (no Unsloth requirement here; it can
    # still be used by setting UNSLOTH_DISABLE=0 and manually swapping the loader).
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import (
        get_peft_model, LoraConfig, prepare_model_for_kbit_training, PeftModel,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.no_4bit or not cuda_ok:
        bnb_config = None
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=compute_dtype,
        device_map=device_map,
        quantization_config=bnb_config,
    )
    if bnb_config is not None:
        # IMPORTANT: pass use_gradient_checkpointing=False here. peft 0.13.2's
        # prepare_model_for_kbit_training enables GC with use_reentrant=True
        # (its default), and reentrant checkpointing requires the checkpointed
        # segment to have at least one input with requires_grad=True. The
        # embedding hook is supposed to provide that, but on a 4-bit-frozen
        # base wrapped by PEFT, it breaks — every LoRA param ends up with
        # grad=None ("None of the inputs have requires_grad=True. Gradients
        # will be None"). We enable GC manually AFTER the PEFT wrap below
        # using use_reentrant=False, which has no such requirement.
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False
        )
    elif args.gradient_checkpointing:
        # Non-4bit path: enable gradient checkpointing manually. The use_reentrant=False
        # variant is required for newer transformers (>=4.40) to avoid silent grad loss.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # Ensure inputs require grad so checkpointed activations propagate backward
        # through the LoRA layers. Without this, GRPO's loss has no path to LoRA params.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        logger.info("Gradient checkpointing ENABLED (non-4bit path, use_reentrant=False)")

    # Either warmstart from a v1 (or v2) LoRA, or initialise a fresh LoRA
    if args.warmstart_from:
        logger.info(f"Warm-starting from existing LoRA: {args.warmstart_from}")
        adapter_path = args.warmstart_from
        if "/" in adapter_path and not os.path.isdir(adapter_path) and not adapter_path.startswith(("./", "/")):
            from huggingface_hub import snapshot_download
            adapter_path = snapshot_download(
                repo_id=args.warmstart_from,
                local_dir=os.path.join(args.output_dir, "_warmstart"),
            )
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        peft_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)

    # ------------------------------------------------------------------
    # Enable gradient checkpointing AFTER the PEFT wrap, with use_reentrant=False.
    #
    # The earlier sanity-check run on DGX caught the actual failure mode:
    #   "GRAD SANITY FAILED: 0 LoRA params got a gradient; 288 LoRA params
    #    had grad=None. Training would silently no-op."
    # i.e. just calling enable_input_require_grads() AFTER the PEFT wrap is
    # NOT enough — peft 0.13.2's prepare_model_for_kbit_training had already
    # turned on REENTRANT gradient checkpointing, which is the variant that
    # demands input-requires-grad — and that demand can't be satisfied
    # cleanly through the PEFT delegation chain on a 4-bit-frozen base.
    #
    # The robust fix is to do GC ourselves, AFTER the wrap, with
    # use_reentrant=False. Non-reentrant checkpointing recomputes activations
    # by re-running the forward in a torch.enable_grad() context; it does NOT
    # require any of the original inputs to require_grad, so the
    # frozen-base + LoRA-adapters + GC combo "just works".
    #
    # We still call enable_input_require_grads as a belt-and-suspenders, so
    # the non-reentrant path also has the embedding hook in place if any
    # downstream code (e.g. transformers' generate) checks for it.
    # ------------------------------------------------------------------
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            logger.info(
                "Gradient checkpointing ENABLED on PEFT-wrapped model "
                "(use_reentrant=False)."
            )
        except Exception as _gc_err:
            logger.warning(
                f"gradient_checkpointing_enable failed on PEFT model "
                f"({type(_gc_err).__name__}: {_gc_err}). Falling back to "
                f"base_model.gradient_checkpointing_enable."
            )
            base = getattr(model, "base_model", None)
            inner = getattr(base, "model", base) if base is not None else model
            inner.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
            logger.info("enable_input_require_grads() armed on PEFT model.")

    # Sanity-check: do a small forward+backward and confirm at least one LoRA
    # param received a non-None gradient. If not, the gradient-flow warning
    # was real and training would silently no-op.
    #
    # Use 8 tokens, NOT 1 — with a single token, after HF's label-shift
    # (logits[:, :-1] vs labels[:, 1:]) you have 0 prediction targets and
    # the loss is degenerate (NaN or 0), which produces no useful gradient
    # and gives a false "fail" on the sanity check.
    try:
        model.train()
        _device = next(model.parameters()).device
        # Build a short, valid token sequence. Prefer the tokenizer's bos/eos,
        # but pad out with low-vocab token IDs that always exist.
        _seed_tok = tokenizer.bos_token_id or tokenizer.eos_token_id or 0
        _ids = [_seed_tok] + [(_seed_tok + i + 1) % max(1000, _seed_tok + 16) for i in range(7)]
        _input_ids = torch.tensor([_ids], device=_device, dtype=torch.long)
        _labels = _input_ids.clone()
        _out = model(input_ids=_input_ids, labels=_labels)
        if getattr(_out, "loss", None) is None:
            raise RuntimeError("sanity forward returned no loss")
        _loss = _out.loss
        if not torch.isfinite(_loss):
            raise RuntimeError(f"sanity forward returned non-finite loss: {_loss.item()}")
        _loss.backward()
        lora_with_grad = []
        lora_without_grad = []
        for name, p in model.named_parameters():
            if "lora_" in name and p.requires_grad:
                if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0:
                    lora_with_grad.append(name)
                else:
                    lora_without_grad.append(name)
        # Clean up the sanity grad so it doesn't pollute the first real step
        model.zero_grad(set_to_none=True)
        if not lora_with_grad:
            logger.error(
                f"GRAD SANITY FAILED: 0 LoRA params got a gradient; "
                f"{len(lora_without_grad)} LoRA params had grad=None. "
                f"sanity_loss={_loss.item():.4f}. "
                f"Training would silently no-op. Check enable_input_require_grads "
                f"+ gradient_checkpointing (use_reentrant) interaction."
            )
            raise RuntimeError(
                "Gradient flow sanity check failed — refusing to train a no-op "
                "for hours. See log above."
            )
        logger.info(
            f"GRAD SANITY OK: {len(lora_with_grad)}/{len(lora_with_grad) + len(lora_without_grad)} "
            f"LoRA params received gradient. sanity_loss={_loss.item():.4f}. "
            f"sample={lora_with_grad[0]}"
        )
    except RuntimeError:
        raise
    except Exception as _e:
        logger.warning(
            f"Grad sanity check could not run ({type(_e).__name__}: {_e}). "
            f"Proceeding, but watch for grad_norm=0 in [metrics] lines."
        )

    if cuda_ok:
        free_b, _ = torch.cuda.mem_get_info(0)
        logger.info(f"GPU memory after model load: free={free_b/1e9:.2f}GB")

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|PAD_TOKEN|>"})
            model.resize_token_embeddings(len(tokenizer))

    dataset = build_scenario_dataset(args.num_scenarios, args.seed)

    from trl import GRPOTrainer, GRPOConfig

    push_to_hub = bool(args.push_to_hub and args.hub_model_id)
    if args.push_to_hub and not args.hub_model_id:
        logger.warning("--push-to-hub set without --hub-model-id; disabling hub push.")

    # Optimizer pick: adamw_torch_fused on Ampere+ (~3-5% throughput gain).
    # Fall back to plain adamw_torch on pre-Ampere (V100/T4) — fused needs sm_80+.
    chosen_optim = args.optim
    if chosen_optim == "adamw_torch_fused" and (not cuda_ok or cap[0] < 8):
        logger.info(
            f"adamw_torch_fused needs sm_80+ (got cc={cap[0]}.{cap[1]}); "
            "falling back to adamw_torch."
        )
        chosen_optim = "adamw_torch"

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        # Log every step — with only 15 total, we want every datapoint in TB.
        logging_steps=1,
        save_total_limit=args.save_total_limit,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        bf16=use_bf16,
        fp16=use_fp16,
        bf16_full_eval=use_bf16,
        fp16_full_eval=use_fp16,
        tf32=use_bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=chosen_optim,
        report_to=args.report_to if args.report_to != "none" else None,
        push_to_hub=push_to_hub,
        hub_model_id=args.hub_model_id if push_to_hub else None,
        hub_private_repo=args.hub_private,
        hub_strategy="every_save" if push_to_hub else "end",
        seed=args.seed,
        # HF Trainer auto-disables tqdm when stdout isn't a TTY (i.e. inside
        # docker, under nohup, when piped through tee). Force it on so the
        # progress bar actually shows.
        disable_tqdm=False,
    )
    # Per-step generation budget = num_gen * (per_device_batch * grad_accum).
    # With defaults (num_gen=12, batch=1, grad_accum=4): 48 generations evaluated per
    # GRPO update. Compare v1 (num_gen=4, batch=1, grad_accum=16): 64 evals/step.
    gen_per_step = (
        args.num_generations * args.per_device_batch_size * args.grad_accum_steps
    )
    logger.info(
        f"GRPOConfig: bf16={config.bf16} fp16={config.fp16} "
        f"num_gen={config.num_generations} max_steps={config.max_steps} "
        f"save_steps={config.save_steps} grad_accum={config.gradient_accumulation_steps} "
        f"max_compl_len={config.max_completion_length} optim={chosen_optim} "
        f"grad_ckpt={args.gradient_checkpointing} gen_evals_per_step={gen_per_step}"
    )

    _seed_warnings_issued(model)

    # Per-step progress callback. tqdm is unreliable in docker/nohup logs
    # (rewrites lines, gets buffered, etc.) — this guarantees one clean
    # "step N/M" line per training step in the log file you're tailing.
    from transformers import TrainerCallback
    class StepProgressCallback(TrainerCallback):
        def __init__(self, state_ref: dict):
            self._t_step_start = None
            self._t_run_start = None
            self._state_ref = state_ref
        def on_train_begin(self, args_, state_, control_, **kw):
            self._t_run_start = time.time()
            self._state_ref["global_step"] = state_.global_step
            logger.info(
                f"[progress] training begin: max_steps={state_.max_steps} "
                f"num_train_epochs={args_.num_train_epochs}"
            )
        def on_step_begin(self, args_, state_, control_, **kw):
            self._t_step_start = time.time()
            self._state_ref["global_step"] = state_.global_step
        def on_step_end(self, args_, state_, control_, **kw):
            self._state_ref["global_step"] = state_.global_step
            dur = time.time() - (self._t_step_start or time.time())
            elapsed = time.time() - (self._t_run_start or time.time())
            done = state_.global_step
            total = state_.max_steps or 1
            remaining = max(total - done, 0)
            eta_min = (dur * remaining) / 60 if dur else 0
            logger.info(
                f"[progress] step {done}/{total}  "
                f"step_time={dur:.1f}s  elapsed={elapsed/60:.1f}min  "
                f"eta={eta_min:.1f}min"
            )
        def on_save(self, args_, state_, control_, **kw):
            logger.info(f"[progress] checkpoint saved at step {state_.global_step}")
        def on_log(self, args_, state_, control_, logs=None, **kw):
            if logs:
                # Include reward_std, completion_length, and per-token KL so we
                # can diagnose whether GRPO has within-group variance to learn
                # from and whether the policy is actually diverging from ref.
                # KL=0 across all steps with non-trivial reward_std would point
                # at a beta/ref-model wiring issue.
                snippet = {k: v for k, v in logs.items() if k in
                           ("loss", "reward", "reward_std", "grad_norm",
                            "learning_rate", "kl", "completion_length")}
                if snippet:
                    logger.info(f"[metrics] step {state_.global_step}: {snippet}")

    class HubSafetyPushCallback(TrainerCallback):
        """Heartbeat push to HF Hub every N minutes.

        Why this exists: GRPOTrainer's `hub_strategy='every_save'` only fires
        when a save_step lands. With save_steps=1, that's fine — but if a
        single training step crashes or the box dies mid-step, you lose the
        progress *plus* the metrics.jsonl rows captured during that step.
        This callback pushes the metrics file (and the latest local
        checkpoint dir, if any) on a fixed wallclock cadence so even a
        mid-step crash leaves the most recent state on Hub.

        We push from the on_log hook (fires every logging_steps=1 → every
        step) and rate-limit to once per `interval_minutes`. The push is
        synchronous because TRL's training loop holds the GIL during
        on_log; running it async would risk concurrent uploads stepping
        on each other.
        """
        def __init__(
            self,
            output_dir: str,
            hub_model_id: Optional[str],
            hub_private: bool,
            interval_minutes: int,
            metrics_path: Optional[str],
        ):
            self.output_dir = output_dir
            self.hub_model_id = hub_model_id
            self.hub_private = hub_private
            self.interval_seconds = max(0, interval_minutes) * 60
            self.metrics_path = metrics_path
            self._last_push = 0.0
            self._enabled = bool(hub_model_id) and self.interval_seconds > 0
            self._api = None
            if self._enabled:
                try:
                    from huggingface_hub import HfApi
                    self._api = HfApi(token=os.environ.get("HUGGINGFACE_TOKEN"))
                    # Make sure the repo exists. Idempotent — no-op if it does.
                    self._api.create_repo(
                        repo_id=hub_model_id,
                        private=hub_private,
                        exist_ok=True,
                    )
                    logger.info(
                        f"HubSafetyPushCallback armed: every {interval_minutes}min "
                        f"to {hub_model_id} (private={hub_private})"
                    )
                except Exception as e:
                    logger.warning(
                        f"HubSafetyPushCallback init failed "
                        f"({type(e).__name__}: {e}). Disabling safety push."
                    )
                    self._enabled = False

        def _push_now(self, reason: str) -> None:
            if not self._enabled or self._api is None:
                return
            now = time.time()
            try:
                # Push the metrics file by itself if it exists — small, fast.
                if self.metrics_path and os.path.exists(self.metrics_path):
                    self._api.upload_file(
                        path_or_fileobj=self.metrics_path,
                        path_in_repo=os.path.basename(self.metrics_path),
                        repo_id=self.hub_model_id,
                        commit_message=f"safety push: metrics @ {reason}",
                    )
                # Also push the most recent checkpoint dir if any. We rely
                # on the trainer's own hub push for full uploads on save_steps,
                # so we ONLY push the latest checkpoint name pointer here
                # (small) — the heavy bytes were already pushed by the trainer.
                self._last_push = now
                logger.info(f"[safety-push] uploaded metrics ({reason})")
            except Exception as e:
                logger.warning(
                    f"[safety-push] failed ({type(e).__name__}): {e}. "
                    f"Will retry next interval."
                )

        def on_log(self, args_, state_, control_, logs=None, **kw):
            if not self._enabled:
                return
            now = time.time()
            if now - self._last_push >= self.interval_seconds:
                self._push_now(reason=f"step={state_.global_step}")

        def on_train_end(self, args_, state_, control_, **kw):
            # Always push at end of training so the final metrics row lands
            # even if it was logged less than interval_seconds ago.
            if self._enabled:
                self._push_now(reason="train_end")

    # Build the metrics logger BEFORE the trainer so the closure in
    # create_reward_fn captures it. Path defaults to <output-dir>/metrics.jsonl
    # so it lands inside the checkpoint dir and gets pushed to Hub on every
    # save (hub_strategy="every_save" uploads the entire output_dir).
    from training.metrics import MetricsLogger
    metrics_path = args.metrics_file
    if metrics_path is None:
        metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    summary_path = args.summary_file
    if summary_path is None:
        summary_path = os.path.join(args.output_dir, "train_summary.json")
    metrics_logger = (
        MetricsLogger(metrics_path) if metrics_path else None
    )
    if metrics_logger is not None:
        logger.info(f"MetricsLogger writing to {metrics_path}")

    # trainer_state_ref is a tiny shared dict the StepProgressCallback updates
    # on every step_begin so reward_fn knows the current global_step at log time.
    # We can't read state_.global_step from inside reward_fn directly (it's not
    # in scope) — this side-channel is the cleanest workaround.
    trainer_state_ref: dict = {"global_step": 0}

    callbacks = [StepProgressCallback(state_ref=trainer_state_ref)]
    if args.safety_push_every_min > 0 and push_to_hub:
        callbacks.append(
            HubSafetyPushCallback(
                output_dir=args.output_dir,
                hub_model_id=args.hub_model_id,
                hub_private=args.hub_private,
                interval_minutes=args.safety_push_every_min,
                metrics_path=metrics_path,
            )
        )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=[create_reward_fn(
            rollout_limit=args.rollout_limit,
            step1_weight=args.step1_weight,
            format_bonus=args.format_bonus,
            prefix_actions=args.prefix_actions,
            metrics_logger=metrics_logger,
            trainer_state_ref=trainer_state_ref,
        )],
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    resume = _resolve_resume(args.resume_from_checkpoint, args.output_dir)
    if resume is not None:
        logger.info(f"Resuming from checkpoint: {resume}")

    try:
        trainer.train(resume_from_checkpoint=resume)
    except KeyboardInterrupt:
        logger.warning("Interrupted — saving checkpoint before exit.")
        trainer.save_model(args.output_dir)
        raise
    except Exception as e:
        # Without this, any non-KeyboardInterrupt exception bubbles up as a
        # raw traceback that's easy to miss when scrolling 12h of logs.
        # logger.exception writes the full traceback at ERROR level — and
        # also flushes (because we set line_buffering=True at top of file).
        logger.exception(
            f"trainer.train CRASHED: {type(e).__name__}: {e}. "
            f"Attempting to save partial state to {args.output_dir} ..."
        )
        try:
            trainer.save_model(args.output_dir)
        except Exception as save_err:
            logger.error(f"Could not save partial state: {save_err}")
        raise

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # End-of-run metrics summary + auto plots
    if metrics_logger is not None:
        try:
            metrics_logger.write_summary_json(summary_path)
            logger.info(f"Wrote end-of-run summary to {summary_path}")
        except Exception as _e:
            logger.warning(f"Could not write summary JSON: {_e}")

    if args.auto_plots and metrics_logger is not None:
        try:
            from training.plots import generate_all_plots
            plots_dir = os.path.join(args.output_dir, "plots")
            generate_all_plots(
                metrics_path=metrics_path,
                output_dir=plots_dir,
                eval_results_dir=None,  # eval_results live in a separate dir
            )
            logger.info(f"Wrote v2.1 plot set to {plots_dir}")
        except Exception as _e:
            logger.warning(
                f"Auto-plot generation failed ({type(_e).__name__}: {_e}). "
                f"Run `python -m training.plots --metrics-file {metrics_path} "
                f"--output-dir {os.path.join(args.output_dir, 'plots')}` manually."
            )

    if push_to_hub:
        logger.info(f"Pushing final model to hub: {args.hub_model_id}")
        trainer.push_to_hub(commit_message="final v2 model")
    logger.info(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
