"""Sweep BalanceConfig variants with tools/calibrate.py, in parallel, logged as one experiment.

    python tools/sweep_balance.py --grid tools/sweeps/w3_single_knobs.json --tag w3-single
    python tools/sweep_balance.py --grid my.json --policies heuristic oracle --n 50 --no-log

Grid file: JSON list of {"name": str, "base": preset (optional), "set": {field: value}}.

Writes (unless --no-log): one experiments.jsonl row per (variant, policy) with
experiment="balance_sweep", one data file research_log/data/<date>_sweep_<tag>.json with
all per-variant summaries (no per-episode records, to keep it small), and one LOG.md entry
with the combined table.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate as C  # noqa: E402


def _run_variant(args):
    variant, policies, n, seed = args
    base = C.get_balance(variant.get("base"))
    ov = C.parse_overrides([f"{k}={json.dumps(v)}" for k, v in variant.get("set", {}).items()])
    cfg = base.with_(**ov) if ov else base
    res = C.calibrate(cfg, policies, n, seed, progress=False)
    for r in res.values():
        r.pop("episodes")
    return variant, cfg.to_dict(), res


def format_sweep(rows) -> str:
    keys = [("survival", "{:.0%}"), ("ep_len", "{:.1f}"), ("a0_life", "{:.1f}"), ("healthy_end", "{:.2f}")]
    policies = list(rows[0][2])
    head = "| variant | " + " | ".join(f"{p} surv / ep_len / A0 / healthy" for p in policies) + " |"
    lines = [head, "|" + "---|" * (len(policies) + 1)]
    for variant, _, res in rows:
        cells = []
        for p in policies:
            m = res[p]["metrics"]
            cells.append(" / ".join(f.format(m[k]) for k, f in keys))
        lines.append(f"| {variant['name']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--policies", nargs="+", default=["random", "heuristic", "oracle"], choices=list(C.POLICIES))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("--note", default="")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args(argv)

    with open(args.grid, encoding="utf-8") as f:
        grid = json.load(f)
    jobs = [(v, args.policies, args.n, args.seed) for v in grid]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(_run_variant, jobs))
    table = format_sweep(rows)
    print(table)
    if args.no_log:
        return rows

    now = dt.datetime.now()
    sha = C.git_sha()
    data_path = C._unique_path(os.path.join(C.LOG_DIR, "data", f"{now:%Y-%m-%d}_sweep_{args.tag}.json"))
    rel = os.path.relpath(data_path, C.LOG_DIR).replace("\\", "/")
    common = {"date": now.isoformat(timespec="seconds"), "git_sha": sha, "experiment": "balance_sweep",
              "tag": args.tag, "n_eps": args.n, "seed": args.seed, "note": args.note,
              "grid_file": os.path.relpath(args.grid, C.ROOT).replace("\\", "/")}
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump({**common, "variants": [{"variant": v, "balance_config": cfg, "results": res}
                                          for v, cfg, res in rows]}, f, indent=1)
    with open(os.path.join(C.LOG_DIR, "experiments.jsonl"), "a", encoding="utf-8") as f:
        for v, cfg, res in rows:
            for p, r in res.items():
                f.write(json.dumps({**common, "variant": v["name"], "balance": v.get("base") or C.DEFAULT_PRESET,
                                    "balance_overrides": v.get("set", {}), "balance_config": cfg,
                                    "policy": p, "metrics": r["metrics"], "death_causes": r["death_causes"],
                                    "healthy_death_causes": r["healthy_death_causes"],
                                    "data_file": rel}) + "\n")
    with open(os.path.join(C.LOG_DIR, "LOG.md"), "a", encoding="utf-8") as f:
        f.write(f"\n---\n\n## {now:%Y-%m-%d %H:%M} — balance sweep `{args.tag}` ({len(rows)} variants)\n\n"
                + (f"{args.note}\n\n" if args.note else "")
                + f"{args.n} episodes per cell, seed {args.seed}, git `{sha}`. Cells: survival / episode length / "
                  f"A0 lifetime / healthy alive at end.\n\n{table}\n\nData: `{rel}`, grid `{common['grid_file']}`.\n")
    print(f"logged -> {os.path.relpath(data_path, C.ROOT)}", file=sys.stderr)
    return rows


if __name__ == "__main__":
    main()
