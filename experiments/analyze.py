"""Where does a checkpoint fail? Greedy play on held-out levels, success broken down by level features.

    python experiments/analyze.py runs/exp/<name>/final.pt [difficulty ...] [--levels 400] [--blocker 0]

Features come from each level's first observation: planets, gravity, planet on the straight line
(blocked), how far the ship must turn to face the goal, distance and goal size. Crashes are split
into walls vs planets using the observation the episode ended on.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from outerworld_env import OuterworldVecEnv, EVAL_SEEDS, TRAIN_SEEDS, _dll   # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import make_net as build_net, load_state   # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("checkpoint")
p.add_argument("difficulties", nargs="*", type=float, default=[0.2, 0.5, 0.8])
p.add_argument("--levels", type=int, default=400)
p.add_argument("--blocker", type=float, default=0.0)
p.add_argument("--sampled", action="store_true", help="sample actions instead of argmax")
p.add_argument("--torch-seed", type=int, default=0, help="seed for --sampled / --unstick sampling")
p.add_argument("--ensemble", default=None,
               help="comma-separated extra checkpoints (same observation setup); act on the mean of all action probabilities")
p.add_argument("--dump", default=None, help="write per-level results to this JSON file ({difficulty: [result, ...]})")
p.add_argument("--unstick", type=int, default=0,
               help="greedy, but sample in any world whose ship has barely moved for this many decisions")
p.add_argument("--train-seeds", action="store_true", help="use training levels instead of held-out ones")
p.add_argument("--keep-slots", type=int, default=None,
               help="zero every obstacle slot after this many (to emulate a smaller observation)")
a = p.parse_args()

ck = torch.load(a.checkpoint, weights_only=False)
cargs = ck.get("args", {})
repeat, rays, ray_range = cargs.get("action_repeat", 1), cargs.get("rays", 0), cargs.get("ray_range", 1.5)
hidden, layers = cargs.get("hidden", 64), cargs.get("layers", 2)
SLOTS = (_dll.ow_obs_size() - 14) // 5   # 8 normally; more with a DLL built with a bigger OBS_MAX_OBSTACLES
obs_size = 14 + 5 * SLOTS + rays


def make_net(out):
    return build_net(obs_size, out, 1.0, hidden, layers, cargs.get("arch", "mlp"), cargs.get("embed", 32),
                     cargs.get("pool", "max"), SLOTS)


policy = make_net(6)
load_state(policy, ck["policy"])

if a.ensemble:
    members = [policy]
    for path in a.ensemble.split(","):
        other = torch.load(path, weights_only=False)
        oargs = other.get("args", {})
        assert (oargs.get("rays", 0), oargs.get("action_repeat", 1), oargs.get("obs_norm", "running")) ==             (rays, repeat, cargs.get("obs_norm", "running")) == (rays, repeat, "none"), f"{path}: different observation setup"
        net = build_net(obs_size, 6, 1.0, oargs.get("hidden", 64), oargs.get("layers", 2), oargs.get("arch", "mlp"),
                        oargs.get("embed", 32), oargs.get("pool", "max"), SLOTS)
        load_state(net, other["policy"])
        members.append(net)

    def policy(x):   # log of the mean probability, so argmax / Categorical(logits=...) work unchanged
        return torch.stack([torch.softmax(m(x), 1) for m in members]).mean(0).log()


def norm(x):
    if a.keep_slots is not None:
        x = x.copy()
        x[:, 14 + 5 * a.keep_slots:14 + 5 * SLOTS] = 0.0
    mode = cargs.get("obs_norm", "running")
    if mode == "none":
        return x.astype(np.float32)
    std = (ck["obs_var"] + 1e-8)**0.5
    if mode == "floor":
        std = np.maximum(std, cargs.get("std_floor", 0.1))
    return np.clip((x - ck["obs_mean"]) / std, -10, 10).astype(np.float32)


S, RS, SHIP_R = 16 / 9, 0.25, 0.04


def position(o):
    """World position of the ship, from the wall-distance features 8 and 10: (pos + half size) / full size."""
    o = np.atleast_2d(o)
    return np.stack([o[:, 8] * 2 * S - S, o[:, 10] * 2.0 - 1.0], axis=1)


def features(obs):
    goal = obs[:, 4:6] * S
    planets = obs[:, 14:14 + 5 * SLOTS].reshape(len(obs), SLOTS, 5)
    present = planets[:, :, 4] > 0.5
    centres, radii = planets[:, :, 0:2] * S, planets[:, :, 2] * RS
    t = np.clip((centres * goal[:, None, :]).sum(2) / (goal**2).sum(1, keepdims=True), 0, 1)
    gap = np.linalg.norm(centres - t[:, :, None] * goal[:, None, :], axis=2) - radii
    return {
        "blocked": ((gap < SHIP_R) & present).any(1),
        "gravity": ((planets[:, :, 3] > 0) & present).any(1),
        "planets": present.sum(1),
        "turn_deg": np.degrees(np.abs(np.arctan2(goal[:, 0], goal[:, 1]))),
        "distance": np.linalg.norm(goal, axis=1),
        "goal_size": obs[:, 6:8].min(1) * S,
    }


def bucket_report(name, values, edges, goals, results):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (values >= lo) & (values < hi)
        if m.sum() >= 5:
            crash = np.mean([results[k] == "crash" for k in np.flatnonzero(m)])
            out.append(f"[{lo:g},{hi:g}): {goals[m].mean():4.0%} (n={m.sum()}, crash {crash:.0%})")
    print(f"  {name:10s} " + "  ".join(out))


print(f"{a.checkpoint}  (iteration {ck.get('iteration')}, repeat {repeat}, rays {rays}, hidden {hidden})")
SEEDS = TRAIN_SEEDS if a.train_seeds else EVAL_SEEDS
dumped = {}
for diff in a.difficulties:
    env = OuterworldVecEnv(a.levels, levels="generated", difficulty=diff, seed_range=SEEDS, shaping="path",
                           action_repeat=repeat, rays=rays, ray_range=ray_range, blocker_chance=a.blocker)
    obs = env.reset(level_seeds=[SEEDS[0] + k for k in range(a.levels)])
    shortest = env._prev_distances.copy()   # shortest route from spawn to the goal, around the planets (distance field)
    flown = np.zeros(a.levels)              # length of the route actually flown (straight segments between decisions)
    prev_pos = position(obs)
    f = features(obs)
    results = [None] * a.levels
    final = np.zeros_like(obs)
    lengths = np.zeros(a.levels)
    torch.manual_seed(a.torch_seed)
    still = np.zeros(a.levels, dtype=int)   # decisions in a row with the ship (almost) not moving
    while None in results:
        with torch.no_grad():
            logits = policy(torch.from_numpy(norm(obs)))
        act = torch.distributions.Categorical(logits=logits).sample() if a.sampled else logits.argmax(1)
        if a.unstick:
            speed = np.hypot(obs[:, 0], obs[:, 1])
            still = np.where(speed < 0.01, still + 1, 0)
            stuck = torch.from_numpy(still >= a.unstick)
            act = torch.where(stuck, torch.distributions.Categorical(logits=logits).sample(), act)
        obs, _, _, _, info = env.step(act.numpy())
        pos = position(obs)
        ended = [ep["index"] for ep in info["episodes"]]
        pos[ended] = position(info["final_obs"][ended])   # those worlds already restarted; use where they ended
        active = np.array([r is None for r in results])
        flown += np.where(active, np.linalg.norm(pos - prev_pos, axis=1), 0.0)
        prev_pos = position(obs)
        for ep in info["episodes"]:
            k = ep["index"]
            if results[k] is None:
                results[k], final[k], lengths[k] = ep["result"], info["final_obs"][k], ep["length"]
    dumped[str(diff)] = results
    goals = np.array([r == "goal" for r in results])
    crashed = np.array([r == "crash" for r in results])
    # wall distances are (pos + W) / 2W etc., so world distance = value * 2W (x) or * 2H (y)
    wall_dist = np.minimum(final[:, 8:10].min(1) * 2 * S, final[:, 10:12].min(1) * 2.0)
    wall_crash = crashed & (wall_dist < SHIP_R + 0.02)
    print(f"\ndifficulty {diff}: success {goals.mean():.0%}  {dict(Counter(results))}  "
          f"crashes into walls {wall_crash.sum()}, planets {crashed.sum() - wall_crash.sum()}  "
          f"mean length (goal) {lengths[goals].mean() if goals.any() else float('nan'):.0f} steps")
    if goals.any():
        eff = flown[goals] / np.maximum(shortest[goals], 1e-6)
        print(f"  path efficiency (flown / shortest route, successes): mean {eff.mean():.2f}, median {np.median(eff):.2f}, "
              f"90th percentile {np.percentile(eff, 90):.2f};  seconds to goal: mean {lengths[goals].mean() / 60:.2f}")
    for key in ("blocked", "gravity"):
        m = f[key]
        print(f"  {key:10s} yes {goals[m].mean() if m.any() else float('nan'):4.0%} (n={m.sum()})   "
              f"no {goals[~m].mean() if (~m).any() else float('nan'):4.0%} (n={(~m).sum()})")
    bucket_report("planets", f["planets"], [0, 2, 4, 6, 9, 11, 13], goals, results)
    bucket_report("turn deg", f["turn_deg"], [0, 30, 60, 90, 135, 181], goals, results)
    bucket_report("distance", f["distance"], [0, 1, 1.5, 2, 2.5, 5], goals, results)
    bucket_report("goal size", f["goal_size"], [0, 0.08, 0.12, 0.16, 1], goals, results)

if a.dump:
    import json
    Path(a.dump).write_text(json.dumps(dumped))
