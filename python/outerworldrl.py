import sys; sys.path.insert(0, "python")
import torch
import gymnasium as gym
import numpy as np
from outerworld_env import OuterworldEnv

env = OuterworldEnv()

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
value_optimizer = torch.optim.Adam(value_net.parameters(), lr=1e-3)

def run_episode():
    obs, info = env.reset()
    probs, rwds, all_obs, actions = [], [], [], []
    terminated = False
    while True:
        all_obs.append(obs)
        prob, action = choose_action(obs)
        obs, reward, terminated, truncated, info = env.step(action.item())
        
        probs.append(prob)
        rwds.append(reward)
        actions.append(action.item())

        if terminated or truncated:
            break
    return probs, rwds, all_obs, obs, terminated, actions, info

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

for i in range(500):
    all_log_probs, all_advantages, all_targets, all_obs, all_actions, successes = [], [], [], [], [], []
    for j in range(10):
        log_probs, rewards, e_obs, obsf, terminated, actions, info = run_episode()
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
        successes.append(info["is_success"])
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
    for epoch in range(10):
        indices = torch.randperm(batch_size)
        fracs = []
        for start in range(0, batch_size, minibatch_size):
            idx = indices[start:start + minibatch_size]
            logits = policy(obs_t[idx])                             
            dist = torch.distributions.Categorical(logits=logits)                             
            new_log_prob = dist.log_prob(actions_t[idx])
            ratio = torch.exp(new_log_prob - old_log_prob[idx])
            surr1 = ratio * advantages_t[idx]
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages_t[idx]
            loss = -torch.min(surr1, surr2).mean() - 0.01 * dist.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            fracs.append(((ratio - 1).abs() > 0.2).float().mean().item())
        clip_frac = np.mean(fracs)

    vloss = torch.tensor(0.0)
    for _ in range(20):
        values = value_net(obs_t).squeeze(1)       # guess again (it improved since the last practice step)
        vloss = ((values - targets_t)**2).mean()
        value_optimizer.zero_grad()
        vloss.backward()
        value_optimizer.step()

    print(f"iteration {i}: success {sum(successes) / len(successes):.0%}, value loss {vloss.item():.1f}, clip frac {clip_frac:.2f}")
    env.save_replay("runs/latest.txt")
    