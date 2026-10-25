"""Prints one line per experiment in runs/exp: progress, curriculum, fixed-difficulty evals.

    python experiments/summarize.py [name filter]
"""

import json
import sys
from pathlib import Path

import numpy as np

flt = sys.argv[1] if len(sys.argv) > 1 else ""


def last(rows, key, n=1):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals[-n:])) if vals else float("nan")


print(f"{'name':34s} {'iter':>5s} {'min':>5s} {'diff':>5s} {'trainS':>6s} | "
      f"{'f0.2':>5s} {'blk':>5s} {'clr':>5s} | {'f0.5':>5s} {'blk':>5s} {'clr':>5s} | {'best':>5s} | final 0/.2/.4/.6/.8/1 | 0+blk")
for d in sorted(Path("runs/exp").iterdir()):
    if flt not in d.name or not (d / "progress.jsonl").exists():
        continue
    rows = [json.loads(l) for l in (d / "progress.jsonl").read_text().splitlines() if l.strip()]
    if not rows:
        continue
    best = max((r["fixed_mean"] for r in rows if "fixed_mean" in r), default=float("nan"))
    line = (f"{d.name:34s} {rows[-1]['i']:5d} {rows[-1]['time'] / 60:5.1f} {rows[-1]['difficulty']:5.2f} "
            f"{last(rows, 'train_success', 20):6.2f} | "
            # last two fixed evals averaged, to smooth noise
            f"{last(rows, 'fixed0.2_success', 2):5.2f} {last(rows, 'fixed0.2_success_blocked', 2):5.2f} "
            f"{last(rows, 'fixed0.2_success_clear', 2):5.2f} | "
            f"{last(rows, 'fixed0.5_success', 2):5.2f} {last(rows, 'fixed0.5_success_blocked', 2):5.2f} "
            f"{last(rows, 'fixed0.5_success_clear', 2):5.2f} | {best:5.2f} |")
    res = d / "results.json"
    if res.exists():
        ev = json.loads(res.read_text())["evals"]
        line += " " + "/".join(f"{ev[k]['success']:.2f}" for k in ("0.0", "0.2", "0.4", "0.6", "0.8", "1.0"))
        line += f" | {ev['0.0_blocker']['success']:.2f}"
    else:
        line += " (running)"
    print(line)
