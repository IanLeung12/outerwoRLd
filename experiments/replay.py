"""Records a checkpoint playing held-out levels, as replay files for the viewer.

    python experiments/replay.py runs/exp/<name>/final.pt --difficulty 1.0 --levels 0-9 --out replays/
    python experiments/replay.py runs/exp/<name>/final.pt --seeds 12345,777 --difficulty 0.8
    python experiments/replay.py runs/exp/<name>/final.pt --random 5
    build/outerworld.exe --replay replays/level0_goal.txt

Plays greedily (argmax). --levels are held-out level indices; --seeds are raw level seeds (the same numbers as
`outerworld --seed N`, so you can fly a level yourself and then watch the agent on it); --random N picks N fresh
seeds from the held-out range. Files are named level<index>_<result>.txt or seed<seed>_<result>.txt.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from outerworld_env import OuterworldVecEnv, EVAL_SEEDS   # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import make_net, load_state   # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("checkpoint")
p.add_argument("--difficulty", type=float, default=1.0)
p.add_argument("--blocker", type=float, default=0.0)
p.add_argument("--levels", default="0-9", help="held-out level indices, e.g. 0-9 or 3,7,12")
p.add_argument("--seeds", default=None, help="raw level seeds, e.g. 12345,777 (overrides --levels)")
p.add_argument("--random", type=int, default=0, help="play N random held-out seeds (overrides --levels)")
p.add_argument("--out", default="replays")
a = p.parse_args()

if a.seeds:
    runs = [(int(x), f"seed{x}") for x in a.seeds.split(",")]
elif a.random:
    runs = [(int(x), f"seed{x}") for x in np.random.default_rng().integers(*EVAL_SEEDS, a.random)]
else:
    if "-" in a.levels:
        lo, hi = map(int, a.levels.split("-"))
        indices = list(range(lo, hi + 1))
    else:
        indices = [int(x) for x in a.levels.split(",")]
    runs = [(EVAL_SEEDS[0] + k, f"level{k}") for k in indices]

ck = torch.load(a.checkpoint, weights_only=False)
cargs = ck.get("args", {})
rays = cargs.get("rays", 0)
policy = make_net(54 + rays, 6, 1.0, cargs.get("hidden", 64), cargs.get("layers", 2),
                  cargs.get("arch", "mlp"), cargs.get("embed", 32), cargs.get("pool", "max"))
load_state(policy, ck["policy"])


def norm(x):
    mode = cargs.get("obs_norm", "running")
    if mode == "none":
        return x.astype(np.float32)
    std = (ck["obs_var"] + 1e-8)**0.5
    if mode == "floor":
        std = np.maximum(std, cargs.get("std_floor", 0.1))
    return np.clip((x - ck["obs_mean"]) / std, -10, 10).astype(np.float32)


out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for seed, label in runs:
    env = OuterworldVecEnv(1, levels="generated", difficulty=a.difficulty, seed_range=EVAL_SEEDS,
                           action_repeat=cargs.get("action_repeat", 1), rays=rays,
                           ray_range=cargs.get("ray_range", 1.5), blocker_chance=a.blocker)
    obs = env.reset(level_seeds=[seed])
    while True:
        with torch.no_grad():
            action = policy(torch.from_numpy(norm(obs))).argmax(1).numpy()
        obs, _, _, _, info = env.step(action)
        if info["episodes"]:
            result = info["episodes"][0]["result"]
            break
    path = out / f"{label}_{result}.txt"
    env.save_replay(str(path))
    print(f"{path}  ({info['episodes'][0]['length']} steps)")
