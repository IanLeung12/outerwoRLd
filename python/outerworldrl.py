import sys; sys.path.insert(0, "python")
import torch
import time
import gymnasium as gym
import numpy as np
from outerworld_env import OuterworldEnv, EVAL_SEEDS
from collections import Counter
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(f"runs/difficulty0_norm_ortho_{time.strftime('%m%d_%H%M')}")


env = OuterworldEnv(levels="generated", difficulty=0.0)
eval_env = OuterworldEnv(levels="generated", difficulty=0.0, seed_range=EVAL_SEEDS)

class RunningMeanStd:
    def __init__(self):
        self.mean = np.zeros(54)
        self.var = np.ones(54)
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
        return np.clip((obs - self.mean) / (self.var + 1e-8)**0.5, -10, 10)

obs_rms = RunningMeanStd()

def init_layer(layer, gain):
    torch.nn.init.orthogonal_(layer.weight, gain)
    torch.nn.init.zeros_(layer.bias)

policy = torch.nn.Sequential(
    torch.nn.Linear(54,64),
    torch.nn.ReLU(),
    torch.nn.Linear(64,64),
    torch.nn.ReLU(),
    torch.nn.Linear(64,6),
)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

value_net = torch.nn.Sequential(
    torch.nn.Linear(54,64),
    torch.nn.ReLU(),
    torch.nn.Linear(64,64),
    torch.nn.ReLU(),
    torch.nn.Linear(64,1),
)
value_optimizer = torch.optim.Adam(value_net.parameters(), lr=3e-4)

for idx in (0, 2):                                
    init_layer(policy[idx], np.sqrt(2))
    init_layer(value_net[idx], np.sqrt(2))
init_layer(policy[4], 0.01)                          
init_layer(value_net[4], 1.0)                        

print(torch.softmax(policy(torch.zeros(54)), dim=0))

def run_episode():
    obs, info = env.reset()
    probs, rwds, all_obs, raw_obs, actions = [], [], [], [], []
    terminated = False
    while True:
        norm_obs = obs_rms.normalize(obs)
        all_obs.append(norm_obs)
        raw_obs.append(obs)
        prob, action = choose_action(norm_obs)
        obs, reward, terminated, truncated, info = env.step(action.item())
        
        probs.append(prob)
        rwds.append(reward)
        actions.append(action.item())

        if terminated or truncated:
            break
    return probs, rwds, all_obs, obs_rms.normalize(obs), terminated, actions, info, raw_obs

def choose_action(obs):
    with torch.no_grad():
        obs = torch.tensor(obs, dtype=torch.float32)
        logits = policy(obs)                                  
        dist = torch.distributions.Categorical(logits=logits)  
        action = dist.sample()                               
        log_prob = dist.log_prob(action)                        

    return log_prob, action


def gae(rewards, values, vfinal, gam, lam):
    advs = [0] * len(rewards)
    running = 0
    for i in range(len(rewards)-1,-1,-1):
        next_value = values[i+1] if i < len(rewards)-1 else vfinal
        delta = rewards[i] + gam * next_value - values[i]
        running = delta + lam * gam * running
        advs[i] = running
    return advs

def evaluate(num_episodes=50):
    successes = 0
    eval_counter = Counter()
    for k in range(num_episodes):
        obs, info = eval_env.reset(options={"level_seed": EVAL_SEEDS[0] + k})
        while True:
            obs = obs_rms.normalize(obs)
            with torch.no_grad():
                obs = torch.tensor(obs, dtype=torch.float32)
                logits = policy(obs)                                   
                action = logits.argmax().item()                            
                obs, reward, terminated, truncated, info = eval_env.step(action)
                if terminated or truncated:
                    successes += info["is_success"]
                    eval_counter[info["result"]] += 1
                    break
    return successes / num_episodes, eval_counter

num_iterations = 500
for i in range(num_iterations):
    frac = 1 - i / num_iterations
    train_counter = Counter()
    for group in optimizer.param_groups:
        group["lr"] = frac * 3e-4
    all_log_probs, all_advantages, all_targets, all_obs, all_actions, all_raw_obs, successes = [], [], [], [], [], [], []
    for j in range(25):
        log_probs, rewards, e_obs, obsf, terminated, actions, info, robs = run_episode()
        train_counter[info["result"]] += 1
        with torch.no_grad():
            obs_t = torch.tensor(np.array(e_obs), dtype=torch.float32)   
            values = value_net(obs_t).squeeze(1).tolist()               

            if terminated:
                vfinal = 0.0                                           
            else:
                final_t = torch.tensor(obsf, dtype=torch.float32)
                vfinal = value_net(final_t).item()                       

        advantages = gae(rewards, values, vfinal, 0.99, 0.95)
        targets = [a + v for a, v in zip(advantages, values)]
        all_log_probs.extend(log_probs)
        all_advantages.extend(advantages)
        all_targets.extend(targets)
        all_obs.extend(e_obs)
        all_actions.extend(actions)
        all_raw_obs.extend(robs)
        successes.append(info["is_success"])
    obs_rms.update(np.array(all_raw_obs))
    log_probs_t = torch.stack(all_log_probs)
    advantages_t = torch.tensor(all_advantages, dtype=torch.float32)
    targets_t = torch.tensor(all_targets, dtype=torch.float32)
    obs_t = torch.tensor(np.array(all_obs), dtype=torch.float32)
    actions_t = torch.tensor(all_actions)

    old_log_prob = log_probs_t
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
    clip_frac = 0.0
    batch_size = len(actions_t)
    minibatch_size = 64
    value_loss = 0.0
    for epoch in range(10):
        indices = torch.randperm(batch_size)
        fracs = []
        vlosses = []
        stop = False
        for start in range(0, batch_size, minibatch_size):
            idx = indices[start:start + minibatch_size]
            logits = policy(obs_t[idx])                             
            dist = torch.distributions.Categorical(logits=logits)                             
            new_log_prob = dist.log_prob(actions_t[idx])
            log_ratio = new_log_prob - old_log_prob[idx]
            ratio = log_ratio.exp()
            approx_kl = ((ratio - 1) - log_ratio).mean().item()
            if approx_kl > 0.02:
                stop = True
                print(f"  stopped at epoch {epoch}, kl {approx_kl:.4f}")
                break
            surr1 = ratio * advantages_t[idx]
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages_t[idx]
            loss = -torch.min(surr1, surr2).mean() - 0.01 * dist.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            fracs.append(((ratio - 1).abs() > 0.2).float().mean().item())

            values = value_net(obs_t[idx]).squeeze(1)
            vloss = ((values - targets_t[idx])**2).mean()
            value_optimizer.zero_grad()
            vloss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
            value_optimizer.step()
            vlosses.append(vloss.item())
        if fracs:
            clip_frac = np.mean(fracs)
            value_loss = np.mean(vlosses)
        if stop:
            break
    
    writer.add_scalar("train/success", sum(successes) / len(successes), i)
    writer.add_scalar("loss/value", value_loss, i)
    writer.add_scalar("ppo/clip_frac", clip_frac, i)
    writer.add_scalar("train/crash", train_counter["crash"] / 25, i)
    writer.add_scalar("train/timeout", train_counter["timeout"] / 25, i)
    print(f"iteration {i}: success {sum(successes) / len(successes):.0%}, value loss {value_loss:.2f}, clip frac {clip_frac:.2f}")
    if i % 10 == 0:
        eval_success, eval_outcomes = evaluate()
        print(f"  eval success: {eval_success:.0%}")
        writer.add_scalar("eval/success", eval_success, i)
        writer.add_scalar("eval/crash", eval_outcomes["crash"] / 50, i)

    env.save_replay("runs/latest.txt")

