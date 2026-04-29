"""Metrics logger for v2.1 training runs.

What this captures (per `reward_fn` call, written as one JSONL row):

  - call index, wall-clock timestamp, training step (if known)
  - n (group size), parse_ok / n
  - reward stats: mean, std, min, max, range
  - cumulative-reward (agent 0) stats: mean, std, min, max
  - rollout length stats (steps after the model's first action)
  - action histogram (top-12 action types with counts)
  - per-rubric averages: aggregated from `obs.metadata.rubric_breakdown`
    on the FIRST step of every rollout — gives a per-step picture of
    which rubrics are firing positively vs negatively
  - episode terminal stats: avg final step, % healthy survived,
    % infected hidden, % vote phases reached

The file is written atomically (tmp + rename) so a crash mid-write never
leaves a half-line behind. Use `MetricsLogger.summary_dict()` for the
end-of-run summary that ships in the Hub repo.

Designed to be imported as a leaf module — no torch / transformers /
trl deps so unit tests can exercise it without the heavyweight stack.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("survivecity_v2.metrics")


@dataclass
class RewardCallStats:
    """Aggregated stats from a single GRPO `reward_fn` call (one batch)."""
    call_idx: int
    wall_time: float
    training_step: Optional[int] = None
    n: int = 0
    parse_ok: int = 0

    # composite reward (the value GRPO normalises within group)
    r_mean: float = 0.0
    r_std: float = 0.0
    r_min: float = 0.0
    r_max: float = 0.0

    # cumulative_rewards[0] across the rollout — un-clipped agent-0 total
    cum_mean: float = 0.0
    cum_std: float = 0.0
    cum_min: float = 0.0
    cum_max: float = 0.0

    # rollout length after the model's first action
    rollout_mean: float = 0.0
    rollout_max: int = 0

    # action_type → count for the model's first action across the group
    actions: dict[str, int] = field(default_factory=dict)

    # per-rubric averages from rubric_breakdown after the model's action
    rubric_means: dict[str, float] = field(default_factory=dict)

    # outcome rates
    n_done: int = 0
    n_healthy_survived: int = 0
    n_infected_hidden: int = 0
    n_reached_vote30: int = 0
    n_reached_vote50: int = 0
    n_reached_vote70: int = 0
    n_reached_vote90: int = 0
    final_step_mean: float = 0.0


class MetricsLogger:
    """Append-only JSONL logger with thread-safe atomic writes.

    Why JSONL: it's append-only friendly, every line is parseable on its
    own, the file is human-readable, and `pandas.read_json(lines=True)`
    loads it in one call. Using one big JSON or pickle would be hostile
    to mid-run crashes — a corrupted last line in JSONL just gets dropped
    instead of taking down the whole file.
    """

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.Lock()
        self._call_idx = 0
        self._records: list[dict] = []  # held in memory for summary
        # If the file exists from a resumed run, count the existing rows so
        # call_idx stays monotonic.
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self._call_idx += 1
                            try:
                                self._records.append(json.loads(line))
                            except json.JSONDecodeError:
                                # Drop a corrupted last line silently — the
                                # file may have been mid-write at crash time.
                                pass
                logger.info(
                    f"MetricsLogger: resumed from existing {self.path} "
                    f"({self._call_idx} prior calls loaded)."
                )
            except OSError as e:
                logger.warning(f"MetricsLogger: could not read existing log: {e}")
                self._call_idx = 0
                self._records = []

    def log_reward_call(
        self,
        rewards: list[float],
        cum_rewards: list[float],
        rollout_lens: list[int],
        action_types: list[str],
        parse_ok: int,
        rubric_breakdowns: Optional[list[dict[str, float]]] = None,
        terminal_summary: Optional[dict[str, Any]] = None,
        training_step: Optional[int] = None,
    ) -> RewardCallStats:
        """Record one reward_fn invocation. Returns the computed stats so
        the caller can also log them through their own logger."""
        with self._lock:
            self._call_idx += 1
            stats = self._compute_stats(
                call_idx=self._call_idx,
                training_step=training_step,
                rewards=rewards,
                cum_rewards=cum_rewards,
                rollout_lens=rollout_lens,
                action_types=action_types,
                parse_ok=parse_ok,
                rubric_breakdowns=rubric_breakdowns,
                terminal_summary=terminal_summary,
            )
            row = stats.__dict__.copy()
            self._records.append(row)
            self._append_jsonl(row)
            return stats

    def _compute_stats(
        self,
        call_idx: int,
        training_step: Optional[int],
        rewards: list[float],
        cum_rewards: list[float],
        rollout_lens: list[int],
        action_types: list[str],
        parse_ok: int,
        rubric_breakdowns: Optional[list[dict[str, float]]],
        terminal_summary: Optional[dict[str, Any]],
    ) -> RewardCallStats:
        n = len(rewards)
        s = RewardCallStats(
            call_idx=call_idx,
            wall_time=time.time(),
            training_step=training_step,
            n=n,
            parse_ok=parse_ok,
        )
        if n > 0:
            s.r_mean = float(statistics.mean(rewards))
            s.r_std = float(statistics.stdev(rewards)) if n > 1 else 0.0
            s.r_min = float(min(rewards))
            s.r_max = float(max(rewards))
        if cum_rewards:
            s.cum_mean = float(statistics.mean(cum_rewards))
            s.cum_std = float(statistics.stdev(cum_rewards)) if len(cum_rewards) > 1 else 0.0
            s.cum_min = float(min(cum_rewards))
            s.cum_max = float(max(cum_rewards))
        if rollout_lens:
            s.rollout_mean = float(statistics.mean(rollout_lens))
            s.rollout_max = int(max(rollout_lens))
        s.actions = dict(Counter(action_types))

        if rubric_breakdowns:
            keys = set()
            for d in rubric_breakdowns:
                keys.update(d.keys())
            for k in keys:
                vals = [d.get(k, 0.0) for d in rubric_breakdowns]
                s.rubric_means[k] = float(statistics.mean(vals))

        if terminal_summary:
            s.n_done = int(terminal_summary.get("n_done", 0))
            s.n_healthy_survived = int(terminal_summary.get("n_healthy_survived", 0))
            s.n_infected_hidden = int(terminal_summary.get("n_infected_hidden", 0))
            s.n_reached_vote30 = int(terminal_summary.get("n_reached_vote30", 0))
            s.n_reached_vote50 = int(terminal_summary.get("n_reached_vote50", 0))
            s.n_reached_vote70 = int(terminal_summary.get("n_reached_vote70", 0))
            s.n_reached_vote90 = int(terminal_summary.get("n_reached_vote90", 0))
            s.final_step_mean = float(terminal_summary.get("final_step_mean", 0.0))
        return s

    def _append_jsonl(self, row: dict) -> None:
        """Atomic-ish append. We append in 'a' mode — POSIX guarantees
        single-line writes are atomic up to PIPE_BUF (4 KB on Linux),
        and our rows are well below that. Larger rows could split, but
        we never write >4 KB per row in practice.
        """
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(row, default=_json_default) + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    # fsync isn't supported on all filesystems (e.g. some
                    # tmpfs setups). It's belt-and-braces, not load-bearing —
                    # the OS will flush on its own schedule.
                    pass
        except OSError as e:
            logger.warning(f"MetricsLogger: append failed ({e}). Row dropped.")

    def summary_dict(self) -> dict[str, Any]:
        """End-of-run aggregate. Use this to write `train_summary.json`."""
        if not self._records:
            return {"n_calls": 0}

        recs = self._records
        n_calls = len(recs)
        first = recs[0]
        last = recs[-1]
        # Reward trajectory headlines
        r_means = [r.get("r_mean", 0.0) for r in recs]
        r_stds = [r.get("r_std", 0.0) for r in recs]
        cum_means = [r.get("cum_mean", 0.0) for r in recs]
        parse_rates = [
            (r.get("parse_ok", 0) / max(1, r.get("n", 1))) for r in recs
        ]
        rollout_means = [r.get("rollout_mean", 0.0) for r in recs]

        # Aggregate action histogram (sum of counters across all calls)
        agg_actions: Counter = Counter()
        for r in recs:
            for k, v in (r.get("actions") or {}).items():
                agg_actions[k] += int(v)

        # Per-rubric averages averaged across calls
        rubric_keys = set()
        for r in recs:
            rubric_keys.update((r.get("rubric_means") or {}).keys())
        rubric_means_avg = {}
        for k in rubric_keys:
            vals = [(r.get("rubric_means") or {}).get(k, 0.0) for r in recs]
            rubric_means_avg[k] = float(statistics.mean(vals))

        return {
            "n_calls": n_calls,
            "first_call_at": first.get("wall_time"),
            "last_call_at": last.get("wall_time"),
            "wallclock_minutes": (
                (last.get("wall_time", 0) - first.get("wall_time", 0)) / 60.0
            ),
            "r_mean_first": r_means[0] if r_means else None,
            "r_mean_last": r_means[-1] if r_means else None,
            "r_mean_overall": float(statistics.mean(r_means)) if r_means else None,
            "r_std_overall": float(statistics.mean(r_stds)) if r_stds else None,
            "cum_mean_overall": float(statistics.mean(cum_means)) if cum_means else None,
            "parse_rate_overall": float(statistics.mean(parse_rates)) if parse_rates else None,
            "rollout_len_overall": float(statistics.mean(rollout_means)) if rollout_means else None,
            "actions_total": dict(agg_actions),
            "rubric_means_avg": rubric_means_avg,
            "metrics_path": self.path,
        }

    def write_summary_json(self, path: str) -> None:
        try:
            with open(path, "w") as f:
                json.dump(self.summary_dict(), f, indent=2, default=_json_default)
        except OSError as e:
            logger.warning(f"MetricsLogger: summary write failed: {e}")

    @property
    def call_count(self) -> int:
        return self._call_idx


def _json_default(o):
    """Make Counter / sets / non-stdlib objects JSON-friendly."""
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if hasattr(o, "__dict__"):
        return {k: v for k, v in o.__dict__.items() if not k.startswith("_")}
    return str(o)


# ---------------------------------------------------------------------------
# Helpers used by train.py to build the terminal-summary dict before logging
# ---------------------------------------------------------------------------

def build_terminal_summary(final_obs_per_completion: list[dict]) -> dict[str, Any]:
    """Aggregate terminal stats across a GRPO group of final observations.

    Inputs: list of `obs.model_dump()` dicts, one per completion in the
    GRPO group, after the rollout finished (whether by `done=True` or by
    hitting rollout_limit).
    """
    n = len(final_obs_per_completion)
    if n == 0:
        return {}
    n_done = sum(1 for o in final_obs_per_completion if o.get("done"))
    n_healthy_survived = sum(
        1 for o in final_obs_per_completion
        if o.get("metadata", {}).get("n_healthy_alive", 0) >= 1
    )
    # Infected hidden: at least one starting infected agent still in
    # latent state at episode end.
    n_infected_hidden = 0
    n_reached_vote30 = 0
    n_reached_vote50 = 0
    n_reached_vote70 = 0
    n_reached_vote90 = 0
    final_steps: list[int] = []
    for o in final_obs_per_completion:
        md = o.get("metadata", {}) or {}
        agents = o.get("agents", []) or []
        starting = set(md.get("starting_infected", []) or [])
        if any(
            a.get("infection_state") == "latent" and a.get("agent_id") in starting
            for a in agents
        ):
            n_infected_hidden += 1
        s = int(o.get("step_count", 0))
        final_steps.append(s)
        if s >= 30:
            n_reached_vote30 += 1
        if s >= 50:
            n_reached_vote50 += 1
        if s >= 70:
            n_reached_vote70 += 1
        if s >= 90:
            n_reached_vote90 += 1
    return {
        "n_done": n_done,
        "n_healthy_survived": n_healthy_survived,
        "n_infected_hidden": n_infected_hidden,
        "n_reached_vote30": n_reached_vote30,
        "n_reached_vote50": n_reached_vote50,
        "n_reached_vote70": n_reached_vote70,
        "n_reached_vote90": n_reached_vote90,
        "final_step_mean": (
            float(statistics.mean(final_steps)) if final_steps else 0.0
        ),
    }
