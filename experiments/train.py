"""Experiment version of python/outerworldrl.py: the same PPO, with every setting a command-line flag.

    python experiments/train.py --name baseline --seed 0
    python experiments/train.py --name repeat4 --action-repeat 4

Writes to runs/exp/<name>/: TensorBoard events, progress.jsonl (one line per iteration),
latest.pt / final.pt checkpoints, and results.json (a final greedy eval on held-out levels
at fixed difficulties, split by whether a planet blocks the straight line to the goal).

Besides the curriculum eval (at the current difficulty), every --fixed-eval-every iterations the
policy is also evaluated at fixed difficulties, so runs with different curricula can be compared.
"""

import argparse
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from outerworld_env import OuterworldVecEnv, EVAL_SEEDS   # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import make_net as build_net   # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--iterations", type=int, default=600)
p.add_argument("--num-envs", type=int, default=32)
p.add_argument("--num-steps", type=int, default=128)          # decisions per world per iteration
p.add_argument("--gamma", type=float, default=0.99)           # per decision
p.add_argument("--lam", type=float, default=0.95)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--anneal-lr", action="store_true", help="decay lr linearly from --lr to 0 over the run")
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--minibatch", type=int, default=256)
p.add_argument("--clip", type=float, default=0.2)
p.add_argument("--ent", type=float, default=0.01)
p.add_argument("--target-kl", type=float, default=0.02)       # 0 = no early stop
p.add_argument("--kl-mode", choices=["minibatch", "epoch"], default="minibatch")
p.add_argument("--hidden", type=int, default=64)
p.add_argument("--layers", type=int, default=2)
p.add_argument("--arch", choices=["mlp", "deepsets"], default="mlp")   # see experiments/models.py
p.add_argument("--embed", type=int, default=32)                         # deepsets: per-planet embedding size
p.add_argument("--pool", choices=["max", "maxmean"], default="max")      # deepsets pooling
p.add_argument("--action-repeat", type=int, default=1)
p.add_argument("--shaping", default="path")
p.add_argument("--progress-scale", type=float, default=1.0)   # multiplies the shaping reward
p.add_argument("--crash-reward", type=float, default=None)    # None = env default (-5)
p.add_argument("--step-penalty", type=float, default=0.0)     # subtracted every decision, so waiting isn't free
p.add_argument("--timeout-terminal", action="store_true")     # don't bootstrap timeouts: running out of time is worth 0
p.add_argument("--start-difficulty", type=float, default=0.2)
p.add_argument("--curriculum", choices=["train", "eval", "none"], default="train")
p.add_argument("--promote-threshold", type=float, default=0.8)
p.add_argument("--promote-step", type=float, default=0.05)
p.add_argument("--blocker-chance", type=float, default=0.0)
# "upto": every episode gets a random difficulty in [0, curriculum level] instead of exactly the level.
# With --curriculum none --start-difficulty 1 that is plain uniform 0..1 (no curriculum at all).
p.add_argument("--sample-difficulty", choices=["off", "upto"], default="off")
p.add_argument("--rays", type=int, default=0)                 # lidar distances appended to the observation
p.add_argument("--ray-range", type=float, default=1.5)
p.add_argument("--eval-episodes", type=int, default=50)
p.add_argument("--fixed-eval-every", type=int, default=25)
p.add_argument("--fixed-eval-difficulties", default="0.2,0.5")
p.add_argument("--fixed-eval-levels", type=int, default=100)
p.add_argument("--final-eval-levels", type=int, default=200)
# running: the user's RunningMeanStd. A feature that never varies in training (e.g. obstacle slot 6 while
# levels have <= 5 planets) gets std 0, so any later value in it blows up to the +-10 clip.
# floor: the same, but the std is never below --std-floor. none: raw observations (already scaled to ~+-1).
p.add_argument("--obs-norm", choices=["running", "floor", "none"], default="running")
p.add_argument("--std-floor", type=float, default=0.1)
p.add_argument("--init-from", default=None)                   # checkpoint to start from
args = p.parse_args()

out_dir = Path("runs/exp") / args.name
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "args.json").write_text(json.dumps(vars(args), indent=1))
from torch.utils.tensorboard import SummaryWriter   # noqa: E402
writer = SummaryWriter(str(out_dir))
progress_file = open(out_dir / "progress.jsonl", "w")

torch.set_num_threads(1)
torch.manual_seed(args.seed)
torch.distributions.Distribution.set_default_validate_args(False)

import outerworld_env   # noqa: E402  (the env reads these module values on every step)
if args.crash_reward is not None:
    outerworld_env.CRASH_REWARD = args.crash_reward
outerworld_env.PROGRESS_REWARD = args.progress_scale

NUM_ENVS, NUM_STEPS = args.num_envs, args.num_steps
env = OuterworldVecEnv(NUM_ENVS, levels="generated", difficulty=args.start_difficulty, shaping=args.shaping,
                       seed=args.seed, action_repeat=args.action_repeat, blocker_chance=args.blocker_chance,
                       rays=args.rays, ray_range=args.ray_range)
OBS = env.obs_size
curriculum_difficulty = args.start_difficulty   # the level the curriculum has reached

if args.sample_difficulty == "upto":
    _start_episode = env._start_episode
    _difficulty_rng = np.random.default_rng(args.seed + 12345)

    def _start_with_random_difficulty(i, level_seed=None):
        env.difficulty = float(_difficulty_rng.uniform(0.0, curriculum_difficulty))
        _start_episode(i, level_seed)

    env._start_episode = _start_with_random_difficulty


class RunningMeanStd:
    def __init__(self):
        self.mean = np.zeros(OBS)
        self.var = np.ones(OBS)
        self.count = 1e-4

    def update(self, batch):
        bmean = batch.mean(axis=0)
        delta = bmean - self.mean
        total = self.count + len(batch)
        self.mean = self.mean + delta * len(batch) / total
        M2 = self.var * self.count + batch.var(axis=0) * len(batch) + delta**2 * self.count * len(batch) / total
        self.var = M2 / total
        self.count = total

    def normalize(self, obs):
        if args.obs_norm == "none":
            return obs.astype(np.float32)
        std = (self.var + 1e-8)**0.5
        if args.obs_norm == "floor":
            std = np.maximum(std, args.std_floor)
        return np.clip((obs - self.mean) / std, -10, 10).astype(np.float32)


obs_rms = RunningMeanStd()


def make_net(out, out_gain):
    return build_net(OBS, out, out_gain, args.hidden, args.layers, args.arch, args.embed, args.pool)


policy = make_net(6, 0.01)
value_net = make_net(1, 1.0)
if args.init_from:
    ck = torch.load(args.init_from, weights_only=False)
    policy.load_state_dict(ck["policy"]); value_net.load_state_dict(ck["value_net"])
    obs_rms.mean, obs_rms.var, obs_rms.count = ck["obs_mean"], ck["obs_var"], ck["obs_count"]
optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
value_optimizer = torch.optim.Adam(value_net.parameters(), lr=args.lr)

POSITION_SCALE, RADIUS_SCALE, SHIP_RADIUS = 16 / 9, 0.25, 0.04


def line_blocked(obs):
    """True for each world where a planet touches the straight line from the ship to the goal centre."""
    goal = obs[:, 4:6] * POSITION_SCALE
    planets = obs[:, 14:54].reshape(len(obs), -1, 5)
    centres = planets[:, :, 0:2] * POSITION_SCALE
    radii = planets[:, :, 2] * RADIUS_SCALE
    t = np.clip((centres * goal[:, None, :]).sum(2) / (goal**2).sum(1, keepdims=True), 0, 1)
    gap = np.linalg.norm(centres - t[:, :, None] * goal[:, None, :], axis=2) - radii
    return ((gap < SHIP_RADIUS) & (planets[:, :, 4] > 0.5)).any(axis=1)


def value_of(norm_obs):
    with torch.no_grad():
        return value_net(torch.from_numpy(norm_obs)).squeeze(1).numpy()


def collect(obs):
    all_obs = np.zeros((NUM_STEPS, NUM_ENVS, OBS), dtype=np.float32)
    raw_obs = np.zeros((NUM_STEPS, NUM_ENVS, OBS), dtype=np.float32)
    actions = np.zeros((NUM_STEPS, NUM_ENVS), dtype=np.int64)
    log_probs = np.zeros((NUM_STEPS, NUM_ENVS), dtype=np.float32)
    values = np.zeros((NUM_STEPS, NUM_ENVS), dtype=np.float32)
    rewards = np.zeros((NUM_STEPS, NUM_ENVS), dtype=np.float32)
    dones = np.zeros((NUM_STEPS, NUM_ENVS), dtype=np.float32)
    episodes = []
    for t in range(NUM_STEPS):
        norm_obs = obs_rms.normalize(obs)
        with torch.no_grad():
            dist = torch.distributions.Categorical(logits=policy(torch.from_numpy(norm_obs)))
            action = dist.sample()
            log_probs[t] = dist.log_prob(action).numpy()
        values[t] = value_of(norm_obs)
        all_obs[t] = norm_obs
        raw_obs[t] = obs
        actions[t] = action.numpy()
        obs, reward, terminated, truncated, info = env.step(actions[t])
        reward = reward - args.step_penalty
        for k in (np.flatnonzero(truncated) if not args.timeout_terminal else []):
            final = obs_rms.normalize(info["final_obs"][k:k + 1])
            reward[k] += args.gamma * value_of(final)[0]
        rewards[t] = reward
        dones[t] = terminated | truncated
        episodes.extend(info["episodes"])
    next_value = value_of(obs_rms.normalize(obs))
    return obs, all_obs, raw_obs, actions, log_probs, values, rewards, dones, next_value, episodes


def gae(rewards, values, dones, next_value):
    advantages = np.zeros_like(rewards)
    running = np.zeros(NUM_ENVS, dtype=np.float32)
    for t in range(NUM_STEPS - 1, -1, -1):
        next_values = values[t + 1] if t < NUM_STEPS - 1 else next_value
        not_done = 1.0 - dones[t]
        delta = rewards[t] + args.gamma * next_values * not_done - values[t]
        running = delta + args.gamma * args.lam * not_done * running
        advantages[t] = running
    return advantages


_eval_envs = {}


def evaluate(difficulty, num_levels, blocker_chance=0.0):
    """Greedy play of num_levels fixed held-out levels. Returns a dict of rates."""
    key = (num_levels, blocker_chance)
    if key not in _eval_envs:
        _eval_envs[key] = OuterworldVecEnv(num_levels, levels="generated", seed_range=EVAL_SEEDS,
                                           action_repeat=args.action_repeat, blocker_chance=blocker_chance,
                                           rays=args.rays, ray_range=args.ray_range)
    eval_env = _eval_envs[key]
    eval_env.difficulty = difficulty
    obs = eval_env.reset(level_seeds=[EVAL_SEEDS[0] + k for k in range(num_levels)])
    blocked = line_blocked(obs)
    results: list[str | None] = [None] * num_levels
    while None in results:
        with torch.no_grad():
            logits = policy(torch.from_numpy(obs_rms.normalize(obs)))
        obs, _, _, _, info = eval_env.step(logits.argmax(dim=1).numpy())
        for ep in info["episodes"]:
            if results[ep["index"]] is None:
                results[ep["index"]] = ep["result"]
    c = Counter(results)
    goals = np.array([r == "goal" for r in results])
    nan = float("nan")
    return {"success": c["goal"] / num_levels, "crash": c["crash"] / num_levels, "timeout": c["timeout"] / num_levels,
            "success_blocked": float(goals[blocked].mean()) if blocked.any() else nan,
            "success_clear": float(goals[~blocked].mean()) if (~blocked).any() else nan,
            "blocked_fraction": float(blocked.mean())}


def save(path, i):
    torch.save({"policy": policy.state_dict(), "value_net": value_net.state_dict(),
                "obs_mean": obs_rms.mean, "obs_var": obs_rms.var, "obs_count": obs_rms.count,
                "iteration": i, "args": vars(args)}, path)


fixed_diffs = [float(d) for d in args.fixed_eval_difficulties.split(",") if d]
start_time = time.time()
l10 = deque(maxlen=10)
best_fixed = -1.0
obs = env.reset()
for i in range(args.iterations):
    t0 = time.time()
    if args.anneal_lr:
        for opt in (optimizer, value_optimizer):
            opt.param_groups[0]["lr"] = args.lr * (1 - i / args.iterations)
    obs, all_obs, raw_obs, actions, log_probs, values, rewards, dones, next_value, episodes = collect(obs)
    t_collect = time.time() - t0
    obs_rms.update(raw_obs.reshape(-1, OBS))
    advantages = gae(rewards, values, dones, next_value)
    targets = advantages + values
    explained_var = 1 - np.var(targets - values) / (np.var(targets) + 1e-8)

    old_log_prob = torch.from_numpy(log_probs.reshape(-1))
    advantages_t = torch.from_numpy(advantages.reshape(-1))
    targets_t = torch.from_numpy(targets.reshape(-1))
    obs_t = torch.from_numpy(all_obs.reshape(-1, OBS))
    actions_t = torch.from_numpy(actions.reshape(-1))
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
    batch_size = len(actions_t)
    clip_frac = value_loss = kl = entropy = 0.0
    epochs_done = args.epochs
    for epoch in range(args.epochs):
        indices = torch.randperm(batch_size)
        fracs, vlosses, kls, entropies = [], [], [], []
        stop = False
        for start in range(0, batch_size, args.minibatch):
            idx = indices[start:start + args.minibatch]
            dist = torch.distributions.Categorical(logits=policy(obs_t[idx]))
            log_ratio = dist.log_prob(actions_t[idx]) - old_log_prob[idx]
            ratio = log_ratio.exp()
            approx_kl = ((ratio - 1) - log_ratio).mean().item()
            kls.append(approx_kl)
            if args.kl_mode == "minibatch" and args.target_kl > 0 and approx_kl > args.target_kl:
                stop = True
                epochs_done = epoch
                break
            surr1 = ratio * advantages_t[idx]
            surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * advantages_t[idx]
            dist_entropy = dist.entropy().mean()
            entropies.append(dist_entropy.item())
            loss = -torch.min(surr1, surr2).mean() - args.ent * dist_entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            fracs.append(((ratio - 1).abs() > args.clip).float().mean().item())

            vloss = ((value_net(obs_t[idx]).squeeze(1) - targets_t[idx])**2).mean()
            value_optimizer.zero_grad()
            vloss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
            value_optimizer.step()
            vlosses.append(vloss.item())
        if fracs:
            clip_frac, value_loss, kl, entropy = np.mean(fracs), np.mean(vlosses), np.mean(kls), np.mean(entropies)
        if args.kl_mode == "epoch" and args.target_kl > 0 and not stop:
            # KL of the whole batch after this epoch, instead of one noisy minibatch
            with torch.no_grad():
                lr_all = torch.distributions.Categorical(logits=policy(obs_t)).log_prob(actions_t) - old_log_prob
                kl = ((lr_all.exp() - 1) - lr_all).mean().item()
            if kl > args.target_kl:
                stop = True
                epochs_done = epoch + 1
        if stop:
            break

    c = Counter(ep["result"] for ep in episodes)
    n = max(len(episodes), 1)
    success = c["goal"] / n
    row = {"i": i, "time": time.time() - start_time, "t_collect": t_collect, "difficulty": curriculum_difficulty,
           "train_success": success, "train_crash": c["crash"] / n, "train_timeout": c["timeout"] / n,
           "episodes": len(episodes), "length": float(np.mean([ep["length"] for ep in episodes])) if episodes else None,
           "value_loss": float(value_loss), "clip_frac": float(clip_frac), "kl": float(kl), "entropy": float(entropy),
           "epochs": epochs_done, "explained_var": float(explained_var)}
    if i % 10 == 0:
        ev = evaluate(curriculum_difficulty, args.eval_episodes, args.blocker_chance)
        row.update({f"eval_{k}": v for k, v in ev.items()})
    if fixed_diffs and (i % args.fixed_eval_every == 0 or i == args.iterations - 1):
        for d in fixed_diffs:
            ev = evaluate(d, args.fixed_eval_levels)
            row.update({f"fixed{d}_{k}": v for k, v in ev.items()})
        mean_fixed = float(np.mean([row[f"fixed{d}_success"] for d in fixed_diffs]))
        row["fixed_mean"] = mean_fixed
        if mean_fixed > best_fixed:
            best_fixed = mean_fixed
            save(out_dir / "best.pt", i)
    for k, v in row.items():
        if k != "i" and v is not None:
            writer.add_scalar(k.replace("_", "/", 1) if k.startswith(("train_", "eval_", "fixed")) else k, v, i)
    progress_file.write(json.dumps(row) + "\n")
    progress_file.flush()
    if i % 50 == 0:
        save(out_dir / "latest.pt", i)
        if env.last_episode is not None:
            env.save_replay(str(out_dir / "latest_replay.txt"))

    if args.curriculum != "none":
        signal = success if args.curriculum == "train" else row.get("eval_success")
        if signal is not None and (episodes or args.curriculum == "eval"):
            l10.append(signal)
        window = l10.maxlen if args.curriculum == "train" else 3   # eval runs every 10 iterations: 3 evals
        recent = list(l10)[-window:]
        if len(recent) >= window and np.mean(recent) > args.promote_threshold:
            curriculum_difficulty = min(curriculum_difficulty + args.promote_step, 1.0)
            if args.sample_difficulty == "off":
                env.difficulty = curriculum_difficulty
            l10.clear()

save(out_dir / "final.pt", args.iterations - 1)
final = {"name": args.name, "args": vars(args), "time_minutes": (time.time() - start_time) / 60,
         "final_difficulty": curriculum_difficulty, "best_fixed_mean": best_fixed, "evals": {}}
for d in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
    final["evals"][str(d)] = evaluate(d, args.final_eval_levels)
final["evals"]["0.0_blocker"] = evaluate(0.0, args.final_eval_levels, 1.0)
(out_dir / "results.json").write_text(json.dumps(final, indent=1))
print(json.dumps({k: v for k, v in final.items() if k != "args"}, indent=1))
