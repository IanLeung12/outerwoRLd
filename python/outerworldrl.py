import sys; sys.path.insert(0, "python")
import torch
import time
import gymnasium as gym
import numpy as np
from outerworld_env import OuterworldVecEnv, EVAL_SEEDS
from collections import Counter, deque
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(f"runs/curriculum_{time.strftime('%m%d_%H%M')}")
torch.set_num_threads(1)
torch.distributions.Distribution.set_default_validate_args(False)

NUM_ENVS = 64          # worlds stepped together (64 beat 32 at equal training time: more levels per update)
NUM_STEPS = 128        # decisions per world per iteration -> NUM_ENVS * NUM_STEPS = 8192 per batch
ACTION_REPEAT = 8      # each decision is held for 8 physics steps (0.13 s), so GAMMA and LAMBDA are per decision
RAYS = 16              # "lidar" distances to the nearest planet or wall, appended to the observation
BLOCKER_CHANCE = 0.5   # share of training levels with a planet across the straight line to the goal
GAMMA = 0.99
LAMBDA = 0.95
EVAL_EPISODES = 50

env = OuterworldVecEnv(NUM_ENVS, levels="generated", difficulty=0.0, shaping="path", seed=0,
                       action_repeat=ACTION_REPEAT, rays=RAYS, blocker_chance=BLOCKER_CHANCE)
# Evaluated on normal levels (no extra blockers), so eval success means the same thing as before
eval_env = OuterworldVecEnv(EVAL_EPISODES, levels="generated", difficulty=0.0, seed_range=EVAL_SEEDS,
                            action_repeat=ACTION_REPEAT, rays=RAYS)
OBS = env.obs_size     # 54 + RAYS

# No observation normalization: the C++ observation is already scaled to about -1..1, and RunningMeanStd broke
# on features that never vary in training. An obstacle slot that was always empty (e.g. slot 6 while levels had
# at most 5 planets) got std 0, so a planet appearing there became +-10 after clipping; and an empty slot 1
# (almost always filled in training) became -9. See experiments/REPORT.md.

def init_layer(layer, gain):
    torch.nn.init.orthogonal_(layer.weight, gain)
    torch.nn.init.zeros_(layer.bias)
    return layer

class PlanetSetNet(torch.nn.Module):
    """Deep Sets: every obstacle slot goes through the same small network (phi), the results are max-pooled
    over the planets that are present, and that summary joins the rest of the observation in a normal MLP.
    Shared weights mean whatever is learned about one planet applies to all of them, in any slot and any number."""

    def __init__(self, out, out_gain, hidden=64, embed=32):
        super().__init__()
        self.phi = torch.nn.Sequential(
            init_layer(torch.nn.Linear(5, embed), np.sqrt(2)), torch.nn.ReLU(),
            init_layer(torch.nn.Linear(embed, embed), np.sqrt(2)), torch.nn.ReLU())
        self.mlp = torch.nn.Sequential(
            init_layer(torch.nn.Linear(OBS - 40 + embed, hidden), np.sqrt(2)), torch.nn.ReLU(),
            init_layer(torch.nn.Linear(hidden, hidden), np.sqrt(2)), torch.nn.ReLU(),
            init_layer(torch.nn.Linear(hidden, out), out_gain))

    def forward(self, obs):
        planets = obs[:, 14:54].reshape(-1, 8, 5)       # 8 slots x (rel x, rel y, radius, density, present)
        present = planets[:, :, 4:5]
        # phi's outputs are >= 0 (ReLU), so zeroing the empty slots leaves the max over the real planets
        pooled = (self.phi(planets) * present).max(dim=1).values
        rest = torch.cat([obs[:, :14], obs[:, 54:]], dim=1)   # ship, goal, walls, gravity, rays
        return self.mlp(torch.cat([rest, pooled], dim=1))

policy = PlanetSetNet(6, 0.01)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
value_net = PlanetSetNet(1, 1.0)
value_optimizer = torch.optim.Adam(value_net.parameters(), lr=3e-4)

print(torch.softmax(policy(torch.zeros(1, OBS)), dim=1))

def value_of(obs):
    with torch.no_grad():
        return value_net(torch.from_numpy(obs)).squeeze(1).numpy()

def collect(obs):
    """Plays NUM_STEPS steps in every world. Episodes that are still running carry on next time."""
    all_obs = np.zeros((NUM_STEPS, NUM_ENVS, OBS), dtype=np.float32)
    actions = np.zeros((NUM_STEPS, NUM_ENVS), dtype=np.int64)
    log_probs = np.zeros((NUM_STEPS, NUM_ENVS), dtype=np.float32)
    values = np.zeros((NUM_STEPS, NUM_ENVS), dtype=np.float32)
    rewards = np.zeros((NUM_STEPS, NUM_ENVS), dtype=np.float32)
    dones = np.zeros((NUM_STEPS, NUM_ENVS), dtype=np.float32)
    episodes = []

    for t in range(NUM_STEPS):
        with torch.no_grad():
            dist = torch.distributions.Categorical(logits=policy(torch.from_numpy(obs)))
            action = dist.sample()
            log_probs[t] = dist.log_prob(action).numpy()
        values[t] = value_of(obs)
        all_obs[t] = obs
        actions[t] = action.numpy()

        obs, reward, terminated, truncated, info = env.step(actions[t])

        # A timeout isn't a real ending: add the value of where the episode was cut off,
        # so from here on every finished episode can be treated the same way
        for k in np.flatnonzero(truncated):
            reward[k] += GAMMA * value_of(info["final_obs"][k:k + 1])[0]
        rewards[t] = reward
        dones[t] = terminated | truncated
        episodes.extend(info["episodes"])

    next_value = value_of(obs)
    return obs, all_obs, actions, log_probs, values, rewards, dones, next_value, episodes

def gae(rewards, values, dones, next_value):
    """Advantages for all worlds at once. dones[t] = the episode ended at step t."""
    advantages = np.zeros_like(rewards)
    running = np.zeros(NUM_ENVS, dtype=np.float32)
    for t in range(NUM_STEPS - 1, -1, -1):
        next_values = values[t + 1] if t < NUM_STEPS - 1 else next_value
        not_done = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * next_values * not_done - values[t]
        running = delta + GAMMA * LAMBDA * not_done * running
        advantages[t] = running
    return advantages

POSITION_SCALE = 16 / 9   # undoes OBS_POSITION_SCALE (constants.h)
RADIUS_SCALE = 0.25       # undoes OBS_RADIUS_SCALE
SHIP_RADIUS = 0.04

def line_blocked(obs):
    """True for each world where a planet touches the straight line from the ship to the goal centre,
    so flying straight at the goal would crash. Uses the (raw) observation only."""
    goal = obs[:, 4:6] * POSITION_SCALE                           # (N, 2)
    planets = obs[:, 14:54].reshape(len(obs), -1, 5)              # (N, 8, 5); the rays come after
    centres = planets[:, :, 0:2] * POSITION_SCALE                 # (N, 8, 2)
    radii = planets[:, :, 2] * RADIUS_SCALE
    # Closest point to each planet centre on the segment ship -> goal, as a fraction t along it
    t = np.clip((centres * goal[:, None, :]).sum(2) / (goal**2).sum(1, keepdims=True), 0, 1)
    gap = np.linalg.norm(centres - t[:, :, None] * goal[:, None, :], axis=2) - radii
    return ((gap < SHIP_RADIUS) & (planets[:, :, 4] > 0.5)).any(axis=1)

def evaluate():
    """Plays each of the EVAL_EPISODES held-out levels once, choosing the most likely action."""
    obs = eval_env.reset(level_seeds=[EVAL_SEEDS[0] + k for k in range(EVAL_EPISODES)])
    blocked = line_blocked(obs)
    results: list[str | None] = [None] * EVAL_EPISODES
    while None in results:
        with torch.no_grad():
            logits = policy(torch.from_numpy(obs))
        obs, reward, terminated, truncated, info = eval_env.step(logits.argmax(dim=1).numpy())
        for ep in info["episodes"]:
            if results[ep["index"]] is None:   # only the first episode on each level counts
                results[ep["index"]] = ep["result"]
    eval_counter = Counter(results)
    goals = np.array([r == "goal" for r in results])
    # Success split by whether a planet is in the way (nan if no level of that kind)
    blocked_success = goals[blocked].mean() if blocked.any() else float("nan")
    clear_success = goals[~blocked].mean() if (~blocked).any() else float("nan")
    return eval_counter["goal"] / EVAL_EPISODES, eval_counter, blocked_success, clear_success, blocked.mean()

num_iterations = 500
l10_suc = deque(maxlen=10)
obs = env.reset()
for i in range(num_iterations):
    obs, all_obs, actions, log_probs, values, rewards, dones, next_value, episodes = collect(obs)
    advantages = gae(rewards, values, dones, next_value)
    targets = advantages + values
    # How much of the spread in returns the value net predicts: 1 = perfect, 0 = no better than a constant
    explained_var = 1 - np.var(targets - values) / (np.var(targets) + 1e-8)

    # Flatten (steps, worlds) into one batch; the update below is the same as before
    log_probs_t = torch.from_numpy(log_probs.reshape(-1))
    advantages_t = torch.from_numpy(advantages.reshape(-1))
    targets_t = torch.from_numpy(targets.reshape(-1))
    obs_t = torch.from_numpy(all_obs.reshape(-1, OBS))
    actions_t = torch.from_numpy(actions.reshape(-1))

    old_log_prob = log_probs_t
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
    clip_frac = 0.0
    batch_size = len(actions_t)
    minibatch_size = 256
    value_loss = 0.0
    kl = 0.0
    entropy = 0.0
    epochs_done = 10   # full epochs completed before the KL early stop
    for epoch in range(10):
        indices = torch.randperm(batch_size)
        fracs = []
        vlosses = []
        kls = []
        entropies = []
        stop = False
        for start in range(0, batch_size, minibatch_size):
            idx = indices[start:start + minibatch_size]
            logits = policy(obs_t[idx])
            dist = torch.distributions.Categorical(logits=logits)
            new_log_prob = dist.log_prob(actions_t[idx])
            log_ratio = new_log_prob - old_log_prob[idx]
            ratio = log_ratio.exp()
            approx_kl = ((ratio - 1) - log_ratio).mean().item()
            kls.append(approx_kl)
            if approx_kl > 0.02:
                stop = True
                epochs_done = epoch
                print(f"  stopped at epoch {epoch}, kl {approx_kl:.4f}")
                break
            surr1 = ratio * advantages_t[idx]
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages_t[idx]
            dist_entropy = dist.entropy().mean()
            entropies.append(dist_entropy.item())
            loss = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            fracs.append(((ratio - 1).abs() > 0.2).float().mean().item())

            values_pred = value_net(obs_t[idx]).squeeze(1)
            vloss = ((values_pred - targets_t[idx])**2).mean()
            value_optimizer.zero_grad()
            vloss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
            value_optimizer.step()
            vlosses.append(vloss.item())
        if fracs:
            clip_frac = np.mean(fracs)
            value_loss = np.mean(vlosses)
            kl = np.mean(kls)
            entropy = np.mean(entropies)
        if stop:
            break

    # Episodes that finished during this batch (it no longer contains a fixed number of them)
    train_counter = Counter(ep["result"] for ep in episodes)
    num_episodes = max(len(episodes), 1)
    successpct = train_counter["goal"] / num_episodes
    writer.add_scalar("train/success", successpct, i)
    writer.add_scalar("loss/value", value_loss, i)
    writer.add_scalar("ppo/clip_frac", clip_frac, i)
    writer.add_scalar("train/crash", train_counter["crash"] / num_episodes, i)
    writer.add_scalar("train/timeout", train_counter["timeout"] / num_episodes, i)
    writer.add_scalar("train/episodes", len(episodes), i)
    if episodes:
        writer.add_scalar("train/length", np.mean([ep["length"] for ep in episodes]), i)
    writer.add_scalar("curriculum/difficulty", env.difficulty, i)
    writer.add_scalar("ppo/approx_kl", kl, i)
    writer.add_scalar("ppo/entropy", entropy, i)
    writer.add_scalar("ppo/epochs", epochs_done, i)
    writer.add_scalar("value/explained_variance", explained_var, i)
    print(f"iteration {i}: success {successpct:.0%} ({len(episodes)} episodes), value loss {value_loss:.2f}, clip frac {clip_frac:.2f}")
    if i % 10 == 0:
        eval_success, eval_outcomes, blocked_success, clear_success, blocked_frac = evaluate()
        print(f"  eval success: {eval_success:.0%} (planet in the way {blocked_success:.0%}, clear {clear_success:.0%})")
        writer.add_scalar("eval/success", eval_success, i)
        writer.add_scalar("eval/crash", eval_outcomes["crash"] / EVAL_EPISODES, i)
        writer.add_scalar("eval/timeout", eval_outcomes["timeout"] / EVAL_EPISODES, i)
        writer.add_scalar("eval/success_blocked", blocked_success, i)
        writer.add_scalar("eval/success_clear", clear_success, i)
        writer.add_scalar("eval/blocked_fraction", blocked_frac, i)
    if i%50 == 0:
        torch.save({
            "policy": policy.state_dict(),
            "value_net": value_net.state_dict(),
            "iteration": i,
            # what experiments/analyze.py and replay.py need to rebuild this network
            "args": {"arch": "deepsets", "rays": RAYS, "action_repeat": ACTION_REPEAT, "obs_norm": "none"},
        }, f"{writer.log_dir}/latest.pt")
    if episodes:
        l10_suc.append(successpct)
    if len(l10_suc) == l10_suc.maxlen and np.mean(l10_suc) > 0.8:
        env.difficulty = min(env.difficulty + 0.05, 1.0)
        eval_env.difficulty = env.difficulty
        l10_suc.clear()
    if env.last_episode is not None:
        env.save_replay("runs/latest.txt")

