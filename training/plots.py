"""Plot generator for v2.1 training runs.

Reads `metrics.jsonl` written by `training.metrics.MetricsLogger` and
optional `eval_results/eval_step_NNNN.json` files, and writes a set of
PNGs to `--output-dir`. Designed for the demo / write-up: every plot is
self-contained, has axis labels, a title, and a sensible default style.

Usage from the CLI (no torch/transformers/trl dependency required):

    python -m training.plots \
        --metrics-file ./checkpoints/metrics.jsonl \
        --eval-results-dir ./eval_results \
        --output-dir ./plots/

The plots:
  1. reward_curve.png         — composite reward (mean ± std) over training calls
  2. cumulative_reward.png    — cumulative_rewards[0] (mean / min / max) over training
  3. parse_rate.png           — parse-success rate over training (sanity check)
  4. rollout_length.png       — heuristic-rollout length distribution + mean over training
  5. action_distribution.png  — model's first-action histogram, evolving over training
  6. rubric_breakdown.png     — per-rubric average contribution per call (stacked area)
  7. survival_rates.png       — % healthy survived / % infected hidden / vote-phase reach
  8. eval_comparison.png      — baseline vs trained, from eval_results/eval_step_*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import statistics
from typing import Any, Optional

logger = logging.getLogger("survivecity_v2.plots")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _load_metrics_jsonl(path: str) -> list[dict[str, Any]]:
    """Load JSONL into a list of dicts. Drops malformed lines silently —
    a corrupted last line from a crash should not block plotting."""
    records: list[dict[str, Any]] = []
    if not os.path.exists(path):
        logger.warning(f"metrics file not found: {path}")
        return records
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _load_eval_results(eval_dir: str) -> list[dict[str, Any]]:
    """Load every eval_step_NNNN.json in the given directory, sorted by step."""
    files = sorted(glob.glob(os.path.join(eval_dir, "eval_step_*.json")))
    out = []
    for fp in files:
        try:
            with open(fp, "r") as f:
                d = json.load(f)
                # Try to grab the step number from the filename if not embedded
                base = os.path.basename(fp)
                if "step" in d:
                    pass
                else:
                    try:
                        # filename is eval_step_<step>.json
                        d["step"] = int(base.split("_")[-1].split(".")[0])
                    except (ValueError, IndexError):
                        d["step"] = None
                out.append(d)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"failed to read eval file {fp}: {e}")
    return out


# ---------------------------------------------------------------------------
# Plot 1 — reward curve over training
# ---------------------------------------------------------------------------

def plot_reward_curve(records: list[dict], out_path: str) -> Optional[str]:
    """Composite reward (mean ± std) over training calls.

    The single most-important graph for the demo: shows whether GRPO is
    actually pushing reward up. A flat line means no learning; a rising
    line is the gold standard for the v2.1 pitch.
    """
    if not records:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r.get("call_idx", i) for i, r in enumerate(records)]
    means = [r.get("r_mean", 0.0) for r in records]
    stds = [r.get("r_std", 0.0) for r in records]
    mins = [r.get("r_min", 0.0) for r in records]
    maxs = [r.get("r_max", 0.0) for r in records]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, means, label="mean reward", color="tab:blue", linewidth=2)
    ax.fill_between(
        xs,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.20, color="tab:blue", label="±1 std (within-group)"
    )
    ax.plot(xs, mins, color="tab:red", linewidth=0.6, alpha=0.6, label="min")
    ax.plot(xs, maxs, color="tab:green", linewidth=0.6, alpha=0.6, label="max")
    ax.set_xlabel("reward_fn call")
    ax.set_ylabel("composite reward (5*step1_raw + cum0 + format_bonus)")
    ax.set_title("v2.1 GRPO composite reward over training")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plot 2 — cumulative reward (agent 0) over training
# ---------------------------------------------------------------------------

def plot_cumulative_reward(records: list[dict], out_path: str) -> Optional[str]:
    """cumulative_rewards[0] across the rollout — the un-clipped agent-0 total.

    Compared to the composite reward (which has the step1_weight + format_bonus
    layered on top), this one shows the raw "did the policy actually do
    something useful in this episode" signal.
    """
    if not records:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r.get("call_idx", i) for i, r in enumerate(records)]
    means = [r.get("cum_mean", 0.0) for r in records]
    mins = [r.get("cum_min", 0.0) for r in records]
    maxs = [r.get("cum_max", 0.0) for r in records]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, means, color="tab:purple", linewidth=2, label="mean cum_reward[0]")
    ax.fill_between(xs, mins, maxs, alpha=0.18, color="tab:purple", label="min/max range")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("reward_fn call")
    ax.set_ylabel("cumulative_rewards[0] (raw, un-clipped)")
    ax.set_title("Agent-0 cumulative raw reward over training")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plot 3 — parse rate over training
# ---------------------------------------------------------------------------

def plot_parse_rate(records: list[dict], out_path: str) -> Optional[str]:
    """parse_ok / n over training — sanity check that the model is producing
    parseable JSON. Should rise to ~100% within a few training calls."""
    if not records:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r.get("call_idx", i) for i, r in enumerate(records)]
    rates = [
        (r.get("parse_ok", 0) / max(1, r.get("n", 1))) for r in records
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, rates, color="tab:cyan", linewidth=2)
    ax.axhline(1.0, color="green", linewidth=0.5, alpha=0.5, label="100%")
    ax.axhline(0.5, color="orange", linewidth=0.5, alpha=0.5, label="50%")
    ax.set_xlabel("reward_fn call")
    ax.set_ylabel("parse-success rate (parse_ok / n)")
    ax.set_title("Action-JSON parse rate over training")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plot 4 — rollout length over training
# ---------------------------------------------------------------------------

def plot_rollout_length(records: list[dict], out_path: str) -> Optional[str]:
    """Heuristic-rollout length (mean + max) over training."""
    if not records:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r.get("call_idx", i) for i, r in enumerate(records)]
    means = [r.get("rollout_mean", 0.0) for r in records]
    maxs = [r.get("rollout_max", 0) for r in records]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, means, color="tab:olive", linewidth=2, label="mean rollout length")
    ax.plot(xs, maxs, color="tab:olive", linewidth=0.7, linestyle="--", alpha=0.7, label="max rollout length")
    ax.set_xlabel("reward_fn call")
    ax.set_ylabel("rollout steps (after model's first action)")
    ax.set_title("Rollout length over training")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plot 5 — action distribution heatmap (action_type evolution)
# ---------------------------------------------------------------------------

def plot_action_distribution(records: list[dict], out_path: str) -> Optional[str]:
    """Stacked bar: model's first-action histogram, evolving over training calls.

    Each column = one reward_fn call. Each colour band = an action type.
    Should show diversification (early) → specialisation as training proceeds.
    """
    if not records:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Collect the union of all action types seen across all calls
    keys: list[str] = []
    seen: set[str] = set()
    for r in records:
        for k in (r.get("actions") or {}).keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    if not keys:
        return None

    # Build a (n_calls, n_actions) matrix of counts, normalised per call
    n_calls = len(records)
    mat = np.zeros((n_calls, len(keys)), dtype=float)
    for i, r in enumerate(records):
        actions = r.get("actions") or {}
        total = sum(actions.values()) or 1
        for j, k in enumerate(keys):
            mat[i, j] = actions.get(k, 0) / total

    xs = list(range(n_calls))
    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(n_calls)
    cmap = plt.get_cmap("tab20", len(keys))
    for j, k in enumerate(keys):
        ax.bar(xs, mat[:, j], bottom=bottom, label=k, color=cmap(j), width=1.0)
        bottom += mat[:, j]
    ax.set_xlabel("reward_fn call")
    ax.set_ylabel("action share (per call)")
    ax.set_title("Model's first-action distribution over training")
    ax.set_ylim(0, 1.001)
    ax.legend(
        loc="center left", bbox_to_anchor=(1.0, 0.5),
        fontsize="small", frameon=False,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plot 6 — per-rubric breakdown (stacked area)
# ---------------------------------------------------------------------------

def plot_rubric_breakdown(records: list[dict], out_path: str) -> Optional[str]:
    """Per-rubric average contribution after the model's first action.

    Two panels: positive contributions (top), negative contributions (bottom).
    Lets you see at a glance which rubrics are paying out and which are
    penalising — useful for tuning magnitudes.
    """
    if not records:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    keys: list[str] = []
    seen: set[str] = set()
    for r in records:
        for k in (r.get("rubric_means") or {}).keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    if not keys:
        return None

    n_calls = len(records)
    mat = np.zeros((n_calls, len(keys)), dtype=float)
    for i, r in enumerate(records):
        rb = r.get("rubric_means") or {}
        for j, k in enumerate(keys):
            mat[i, j] = float(rb.get(k, 0.0))

    xs = list(range(n_calls))
    fig, (ax_pos, ax_neg) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    cmap = plt.get_cmap("tab20", len(keys))
    pos_bottom = np.zeros(n_calls)
    neg_bottom = np.zeros(n_calls)
    for j, k in enumerate(keys):
        col = cmap(j)
        pos_part = np.where(mat[:, j] > 0, mat[:, j], 0.0)
        neg_part = np.where(mat[:, j] < 0, mat[:, j], 0.0)
        if pos_part.any():
            ax_pos.bar(xs, pos_part, bottom=pos_bottom, label=k, color=col, width=1.0)
            pos_bottom += pos_part
        if neg_part.any():
            ax_neg.bar(xs, neg_part, bottom=neg_bottom, label=k, color=col, width=1.0)
            neg_bottom += neg_part
    ax_pos.set_ylabel("positive rubric contributions")
    ax_neg.set_ylabel("negative rubric contributions")
    ax_neg.set_xlabel("reward_fn call")
    ax_pos.set_title("Per-rubric mean contribution (model's first action)")
    ax_pos.grid(True, alpha=0.3)
    ax_neg.grid(True, alpha=0.3)
    ax_pos.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
                  fontsize="small", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plot 7 — survival rates over training
# ---------------------------------------------------------------------------

def plot_survival_rates(records: list[dict], out_path: str) -> Optional[str]:
    """% rollouts where (a) at least one healthy survived, (b) at least one
    infected stayed hidden, (c-e) episode reached vote phases 30/60/90.
    """
    if not records:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r.get("call_idx", i) for i, r in enumerate(records)]

    def rate(key: str) -> list[float]:
        return [
            (r.get(key, 0) / max(1, r.get("n", 1))) for r in records
        ]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_top.plot(xs, rate("n_healthy_survived"), label="healthy team survived",
                color="tab:green", linewidth=2)
    ax_top.plot(xs, rate("n_infected_hidden"), label="infected stayed hidden",
                color="tab:red", linewidth=2)
    ax_top.set_ylabel("rate (per rollout in group)")
    ax_top.set_ylim(-0.02, 1.02)
    ax_top.set_title("Episode outcomes over training")
    ax_top.legend(loc="upper right")
    ax_top.grid(True, alpha=0.3)

    ax_bot.plot(xs, rate("n_reached_vote30"), label="reached vote 1 (t=30)", linewidth=1.5)
    ax_bot.plot(xs, rate("n_reached_vote60"), label="reached vote 2 (t=60)", linewidth=1.5)
    ax_bot.plot(xs, rate("n_reached_vote90"), label="reached vote 3 (t=90)", linewidth=1.5)
    ax_bot.set_ylabel("rate (per rollout in group)")
    ax_bot.set_ylim(-0.02, 1.02)
    ax_bot.set_xlabel("reward_fn call")
    ax_bot.legend(loc="upper right")
    ax_bot.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plot 8 — eval comparison (baseline vs trained)
# ---------------------------------------------------------------------------

def plot_eval_comparison(eval_records: list[dict], out_path: str) -> Optional[str]:
    """Baseline vs trained metrics from `training/eval.py` output JSONs."""
    if not eval_records:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Use the LATEST eval file for the baseline-vs-trained comparison
    last = eval_records[-1]
    baseline = last.get("baseline_aggregates") or last.get("baseline") or {}
    trained = last.get("trained_aggregates") or last.get("trained") or {}

    metrics = [
        ("survival_rate", "Healthy team survived"),
        ("avg_steps", "Avg episode length"),
        ("avg_reward", "Avg cumulative reward"),
        ("vote_accuracy", "Vote accuracy (any phase)"),
        ("medication_roi", "Medication ROI"),
    ]
    keys = [m[0] for m in metrics if m[0] in baseline or m[0] in trained]
    if not keys:
        return None

    bvals = [float(baseline.get(k, 0.0) or 0.0) for k in keys]
    tvals = [float(trained.get(k, 0.0) or 0.0) for k in keys]

    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, bvals, width, label="baseline (random)", color="tab:gray")
    ax.bar(x + width / 2, tvals, width, label="trained", color="tab:blue")
    ax.set_xticks(x)
    labels_pretty = {m[0]: m[1] for m in metrics}
    ax.set_xticklabels([labels_pretty.get(k, k) for k in keys], rotation=15, ha="right")
    ax.set_title(f"Eval comparison — {os.path.basename(last.get('source_file', 'latest'))}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Top-level: generate the full plot set
# ---------------------------------------------------------------------------

def generate_all_plots(
    metrics_path: str,
    output_dir: str,
    eval_results_dir: Optional[str] = None,
) -> dict[str, Optional[str]]:
    """Generate all 8 plots into `output_dir`. Returns a dict mapping plot
    name → output PNG path (or None if skipped due to missing data).
    """
    os.makedirs(output_dir, exist_ok=True)
    records = _load_metrics_jsonl(metrics_path)
    if not records:
        logger.warning(
            "No metrics records loaded — skipping training-curve plots. "
            "Did the run complete at least one reward_fn call?"
        )

    out: dict[str, Optional[str]] = {}
    out["reward_curve"] = plot_reward_curve(
        records, os.path.join(output_dir, "reward_curve.png"))
    out["cumulative_reward"] = plot_cumulative_reward(
        records, os.path.join(output_dir, "cumulative_reward.png"))
    out["parse_rate"] = plot_parse_rate(
        records, os.path.join(output_dir, "parse_rate.png"))
    out["rollout_length"] = plot_rollout_length(
        records, os.path.join(output_dir, "rollout_length.png"))
    out["action_distribution"] = plot_action_distribution(
        records, os.path.join(output_dir, "action_distribution.png"))
    out["rubric_breakdown"] = plot_rubric_breakdown(
        records, os.path.join(output_dir, "rubric_breakdown.png"))
    out["survival_rates"] = plot_survival_rates(
        records, os.path.join(output_dir, "survival_rates.png"))

    if eval_results_dir:
        eval_records = _load_eval_results(eval_results_dir)
        out["eval_comparison"] = plot_eval_comparison(
            eval_records, os.path.join(output_dir, "eval_comparison.png"))
    else:
        out["eval_comparison"] = None

    for k, v in out.items():
        if v:
            logger.info(f"  wrote {k:<22s} -> {v}")
        else:
            logger.info(f"  skipped {k} (no data)")
    return out


def main():
    p = argparse.ArgumentParser(
        description="Generate the v2.1 training plot set from metrics.jsonl"
    )
    p.add_argument(
        "--metrics-file", required=True,
        help="Path to metrics.jsonl (written by training.metrics.MetricsLogger).")
    p.add_argument(
        "--eval-results-dir", default=None,
        help="Optional directory containing eval_step_NNNN.json files. If "
             "given, the eval-comparison plot is generated.")
    p.add_argument(
        "--output-dir", default="./plots",
        help="Where to write the PNGs. Created if missing.")
    args = p.parse_args()
    generate_all_plots(
        metrics_path=args.metrics_file,
        output_dir=args.output_dir,
        eval_results_dir=args.eval_results_dir,
    )


if __name__ == "__main__":
    main()
