"""Draws experiments/progress.png: held-out success vs difficulty, one line per step of the night's improvements.

    python experiments/plot_progress.py

Each line is the mean of the listed runs' results.json (greedy, 200 held-out levels per difficulty, 600 iterations).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

STEPS = [   # label, runs (averaged over the seeds that exist)
    ("Before (your setup)", ["r1_baseline"]),
    ("+ repeat 4, blocker levels, rays", ["r3_rep4_rays16_mix50_d0", "r3_combo_seed1"]),
    ("+ no observation normalization", ["r4_combo_nonorm", "r4_combo_nonorm_seed1"]),
    ("+ repeat 8, Deep Sets", ["r10_ds_rep8", "r10_ds_rep8_seed1"]),
    ("+ 64 worlds, 2x data per iteration", ["r12_ds_rep8_envs64", "r12_ds_rep8_envs64_seed1", "r12_ds_rep8_envs64_seed2"]),
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # dataviz reference palette, slots 1-5
DIFFS = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
x = [float(d) for d in DIFFS]
for (label, runs), color in zip(STEPS, COLORS):
    found = [json.loads(p.read_text()) for r in runs if (p := Path("runs/exp") / r / "results.json").exists()]
    if not found:
        continue
    y = [100 * sum(f["evals"][d]["success"] for f in found) / len(found) for d in DIFFS]
    seeds = f" ({len(found)} seeds)" if len(found) > 1 else ""
    ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5, label=label + seeds,
            markeredgecolor=SURFACE, markeredgewidth=1.5, solid_capstyle="round")
    ax.annotate(f"{y[-1]:.0f}%", (1.0, y[-1]), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)

ax.set_xlim(-0.03, 1.08)
ax.set_ylim(0, 100)
ax.set_xticks(x)
ax.set_xlabel("Level difficulty", color=INK2, fontsize=10)
ax.set_ylabel("Success on held-out levels (%)", color=INK2, fontsize=10)
ax.set_title("Held-out success after 600 training iterations", color=INK, fontsize=12, loc="left", pad=12)
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9, length=0)
leg = ax.legend(loc="lower left", fontsize=8.5, frameon=False, labelcolor=INK)
fig.tight_layout()
out = Path(__file__).resolve().parent / "progress.png"
fig.savefig(out, facecolor=SURFACE)
print(out)
