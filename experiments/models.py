"""Networks for experiments/train.py (and analyze.py, which must build the same ones).

mlp:      the user's network: observation -> hidden layers -> output.
deepsets: each obstacle slot (rel x, rel y, radius, density, present) goes through the SAME small network phi,
          the results are max-pooled over the slots that are present, and that summary is concatenated with
          the rest of the observation (ship state, goal, walls, gravity, rays) before the usual MLP.
          Shared weights mean what is learned about one planet applies to every planet, and max pooling
          doesn't care about slot order or how many planets there are.
"""

import numpy as np
import torch

BASE = 14                        # ship / goal / walls / gravity values before the obstacle slots
SLOTS, PER_SLOT = 8, 5


def _init(lin, gain):
    torch.nn.init.orthogonal_(lin.weight, gain)
    torch.nn.init.zeros_(lin.bias)
    return lin


def _mlp(size_in, hidden, layers, out, out_gain):
    mods, size = [], size_in
    for _ in range(layers):
        mods += [_init(torch.nn.Linear(size, hidden), np.sqrt(2)), torch.nn.ReLU()]
        size = hidden
    mods.append(_init(torch.nn.Linear(size, out), out_gain))
    return torch.nn.Sequential(*mods)


class DeepSets(torch.nn.Module):
    def __init__(self, obs_size, out, out_gain, hidden, layers, embed, pool="max", slots=SLOTS):
        super().__init__()
        self.slots = slots   # obstacle slots in the observation; the weights don't depend on it
        self.pool = pool   # "max", or "maxmean": max and mean over the present planets, concatenated
        self.extra = obs_size - BASE - slots * PER_SLOT   # e.g. rays
        self.phi = torch.nn.Sequential(
            _init(torch.nn.Linear(PER_SLOT, embed), np.sqrt(2)), torch.nn.ReLU(),
            _init(torch.nn.Linear(embed, embed), np.sqrt(2)), torch.nn.ReLU())
        pooled_size = embed * (2 if pool == "maxmean" else 1)
        self.rho = _mlp(BASE + self.extra + pooled_size, hidden, layers, out, out_gain)

    def forward(self, x):
        end = BASE + self.slots * PER_SLOT
        slots = x[:, BASE:end].reshape(-1, self.slots, PER_SLOT)
        present = slots[:, :, 4:5]
        # phi outputs are >= 0 (ReLU), so zeroing the empty slots leaves the max over real planets
        embedded = self.phi(slots) * present
        pooled = embedded.max(dim=1).values
        if self.pool == "maxmean":
            mean = embedded.sum(dim=1) / present.sum(dim=1).clamp(min=1.0)
            pooled = torch.cat([pooled, mean], dim=1)
        rest = torch.cat([x[:, :BASE], x[:, end:]], dim=1)
        return self.rho(torch.cat([rest, pooled], dim=1))


def make_net(obs_size, out, out_gain, hidden=64, layers=2, arch="mlp", embed=32, pool="max", slots=SLOTS):
    if arch == "deepsets":
        return DeepSets(obs_size, out, out_gain, hidden, layers, embed, pool, slots)
    return _mlp(obs_size, hidden, layers, out, out_gain)


def load_state(net, state):
    """Loads a state dict, also accepting python/outerworldrl.py's PlanetSetNet (which names rho "mlp")."""
    if isinstance(net, DeepSets):
        state = {("rho." + k[4:] if k.startswith("mlp.") else k): v for k, v in state.items()}
    net.load_state_dict(state)
