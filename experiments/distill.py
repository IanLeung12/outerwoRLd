"""Distills an ensemble of policies into one network (DAgger-style), on training levels only.

    python experiments/distill.py --name d1 --teachers a/final.pt,b/final.pt [--init a/final.pt | --fresh]

Each round: the student plays (sampling its own actions, so it visits the states *it* gets into, including its
mistakes), the teachers label every visited state with their mean action probabilities, those go into a buffer,
and the student is trained to match them (cross-entropy with soft targets = KL(teacher || student) + const).
Writes runs/exp/<name>/final.pt in the same format as train.py, so analyze.py / replay.py can load it.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from outerworld_env import OuterworldVecEnv, EVAL_SEEDS, TRAIN_SEEDS   # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import make_net, load_state   # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--teachers", required=True, help="comma-separated checkpoints")
p.add_argument("--init", default=None, help="start the student from this checkpoint (default: the first teacher)")
p.add_argument("--fresh", action="store_true", help="start the student from random weights instead")
p.add_argument("--rounds", type=int, default=150)
p.add_argument("--num-envs", type=int, default=64)
p.add_argument("--num-steps", type=int, default=128)
p.add_argument("--difficulties", default="0.5,1.0", help="the worlds are split evenly between these")
p.add_argument("--blocker-chance", type=float, default=0.5)
p.add_argument("--buffer", type=int, default=200_000)
p.add_argument("--updates", type=int, default=100, help="minibatch updates per round")
p.add_argument("--minibatch", type=int, default=1024)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--eval-every", type=int, default=25)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

torch.manual_seed(args.seed)
out = Path("runs/exp") / args.name
out.mkdir(parents=True, exist_ok=True)
(out / "args.json").write_text(json.dumps(vars(args), indent=1))


def load(path):
    ck = torch.load(path, weights_only=False)
    a = ck.get("args", {})
    net = make_net(54 + a.get("rays", 0), 6, 1.0, a.get("hidden", 64), a.get("layers", 2), a.get("arch", "mlp"),
                   a.get("embed", 32), a.get("pool", "max"))
    load_state(net, ck["policy"])
    return ck, net


teacher_cks, teachers = zip(*(load(t) for t in args.teachers.split(",")))
cargs = teacher_cks[0]["args"]
for ck in teacher_cks:
    a = ck["args"]
    assert (a.get("rays"), a.get("action_repeat"), a.get("obs_norm")) == \
        (cargs.get("rays"), cargs.get("action_repeat"), "none"), "teachers need the same raw observation setup"
REPEAT, RAYS, RAY_RANGE = cargs["action_repeat"], cargs["rays"], cargs.get("ray_range", 1.5)
OBS = 54 + RAYS

init_ck, student = load(args.init or args.teachers.split(",")[0])
if args.fresh:
    student = make_net(OBS, 6, 0.01, cargs.get("hidden", 64), cargs.get("layers", 2), cargs.get("arch", "mlp"),
                       cargs.get("embed", 32), cargs.get("pool", "max"))
optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)


def teacher_probs(x):
    with torch.no_grad():
        return torch.stack([torch.softmax(t(x), 1) for t in teachers]).mean(0)


diffs = [float(d) for d in args.difficulties.split(",")]
per = args.num_envs // len(diffs)
envs = [OuterworldVecEnv(per, levels="generated", difficulty=d, seed_range=TRAIN_SEEDS, seed=args.seed * 100 + k,
                         action_repeat=REPEAT, rays=RAYS, ray_range=RAY_RANGE, blocker_chance=args.blocker_chance,
                         shaping=cargs.get("shaping", "path"))
        for k, d in enumerate(diffs)]
obs = [e.reset() for e in envs]


def evaluate(diff, levels=200):
    env = OuterworldVecEnv(levels, levels="generated", difficulty=diff, seed_range=EVAL_SEEDS,
                           action_repeat=REPEAT, rays=RAYS, ray_range=RAY_RANGE)
    o = env.reset(level_seeds=[EVAL_SEEDS[0] + k for k in range(levels)])
    results = [None] * levels
    while None in results:
        with torch.no_grad():
            o, _, _, _, info = env.step(student(torch.from_numpy(o)).argmax(1).numpy())
        for ep in info["episodes"]:
            if results[ep["index"]] is None:
                results[ep["index"]] = ep["result"]
    env.close()
    return sum(r == "goal" for r in results) / levels


buf_obs = torch.zeros((args.buffer, OBS))
buf_probs = torch.zeros((args.buffer, 6))
filled = cursor = 0
log = open(out / "progress.jsonl", "w")
for rnd in range(args.rounds + 1):
    if rnd % args.eval_every == 0:
        row = {"round": rnd, **{f"eval/{d}": evaluate(d) for d in (0.5, 1.0)}}
        print(row, flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()
    if rnd == args.rounds:
        break
    t0 = time.time()
    successes = episodes = 0
    for _ in range(args.num_steps):
        for k, env in enumerate(envs):
            x = torch.from_numpy(obs[k])
            n = len(x)
            idx = (torch.arange(n) + cursor) % args.buffer
            buf_obs[idx], buf_probs[idx] = x, teacher_probs(x)
            cursor = (cursor + n) % args.buffer
            filled = min(filled + n, args.buffer)
            with torch.no_grad():
                act = torch.distributions.Categorical(logits=student(x)).sample().numpy()
            obs[k], _, _, _, info = env.step(act)
            for ep in info["episodes"]:
                episodes += 1
                successes += ep["result"] == "goal"
    losses = []
    for _ in range(args.updates):
        mb = torch.randint(0, filled, (args.minibatch,))
        logp = torch.log_softmax(student(buf_obs[mb]), 1)
        loss = -(buf_probs[mb] * logp).sum(1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if rnd % 5 == 0:
        print(f"round {rnd}: loss {np.mean(losses):.4f}  student train success {successes / max(episodes, 1):.2f}"
              f"  ({time.time() - t0:.1f}s)", flush=True)

torch.save({"policy": student.state_dict(), "value_net": init_ck.get("value_net"), "iteration": args.rounds,
            "args": {**cargs, "distilled_from": args.teachers}}, out / "final.pt")
print("saved", out / "final.pt")
