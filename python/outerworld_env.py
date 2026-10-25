"""Gymnasium environment for Outerworld.

The physics runs in C++ (outerworld_py.dll, built by CMake). This file wraps it in the
standard Gymnasium interface and defines the reward, so the reward can be changed
without recompiling.

DLL location: $OUTERWORLD_DLL if set, otherwise <repo>/build/outerworld_py.dll.
"""

import ctypes
import os
from pathlib import Path

import gymnasium as gym
import numpy as np

# Reward (see the reward discussion: terminal rewards + potential-based progress shaping)
GOAL_REWARD = 10.0
CRASH_REWARD = -5.0
PROGRESS_REWARD = 1.0        # per unit of distance closer to the goal (see the shaping option)
MAX_STEPS = 1200             # 20 seconds at 60 steps/s, then the episode is truncated

# Generated-level seeds. Train on one range and evaluate on the other: the eval levels are ones
# the agent has never seen, so the gap between the two success rates measures memorization.
TRAIN_SEEDS = (0, 1_000_000)
EVAL_SEEDS = (1_000_000_000, 1_001_000_000)

_RUNNING, _CRASHED, _REACHED_GOAL = 0, 1, 2


def _load_dll():
    default = Path(__file__).resolve().parent.parent / "build" / "outerworld_py.dll"
    path = Path(os.environ.get("OUTERWORLD_DLL", default))
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - build the project first (cmake --build build)")
    dll = ctypes.CDLL(str(path))

    world_p = ctypes.c_void_p
    dll.ow_obs_size.restype = ctypes.c_int
    dll.ow_num_actions.restype = ctypes.c_int
    dll.ow_create.restype = world_p
    dll.ow_destroy.argtypes = [world_p]
    dll.ow_load_test_level.argtypes = [world_p]
    dll.ow_load_generated.argtypes = [world_p, ctypes.c_uint, ctypes.c_float]
    dll.ow_reset.argtypes = [world_p]
    dll.ow_step.argtypes = [world_p, ctypes.c_int]
    dll.ow_step.restype = ctypes.c_int
    dll.ow_observe.argtypes = [world_p, ctypes.POINTER(ctypes.c_float)]
    dll.ow_goal_distance.argtypes = [world_p]
    dll.ow_goal_distance.restype = ctypes.c_float
    dll.ow_path_distance.argtypes = [world_p]
    dll.ow_path_distance.restype = ctypes.c_float
    dll.ow_step_count.argtypes = [world_p]
    dll.ow_step_count.restype = ctypes.c_int
    return dll


_dll = _load_dll()


class OuterworldEnv(gym.Env):
    """One ship, one level. Actions 0-5 = {no thrust, thrust} x {no turn, left, right}.

    levels="test":      always the hand-made test level (the default)
    levels="generated": a new random level every episode, with its seed drawn from seed_range.
                        env.difficulty (0 = easy .. 1 = hard) can be changed between episodes,
                        e.g. for a curriculum. reset(options={"level_seed": n}) plays one exact level.

    shaping chooses the progress reward:
        "straight": reward for getting closer in a straight line (the default)
        "path":     reward for getting closer by the route around obstacles (the distance field)
        "none":     no progress reward, only the goal and crash rewards

    info["result"] at the end of an episode is "goal", "crash" or "timeout",
    info["is_success"] is True when the goal was reached, and info["level"] names the level.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps=MAX_STEPS, levels="test", difficulty=0.5, seed_range=TRAIN_SEEDS, shaping="straight"):
        super().__init__()
        if levels not in ("test", "generated"):
            raise ValueError(f'levels must be "test" or "generated", not {levels!r}')
        if shaping not in ("straight", "path", "none"):
            raise ValueError(f'shaping must be "straight", "path" or "none", not {shaping!r}')
        self.shaping = shaping
        self.max_steps = max_steps
        self.levels = levels
        self.difficulty = difficulty
        self.seed_range = seed_range
        self.level = "test"         # the current level's name, as written in replays
        obs_size = _dll.ow_obs_size()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(_dll.ow_num_actions())

        self._world = _dll.ow_create()
        self._obs = np.zeros(obs_size, dtype=np.float32)
        self._obs_ptr = self._obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._prev_distance = 0.0
        self._actions = []          # actions of the current episode, for replays
        self.last_episode = None    # (level, actions) of the most recently finished episode

    def _observe(self):
        _dll.ow_observe(self._world, self._obs_ptr)
        return self._obs.copy()

    def _distance(self):
        if self.shaping == "path":
            return _dll.ow_path_distance(self._world)
        if self.shaping == "straight":
            return _dll.ow_goal_distance(self._world)
        return 0.0   # "none": the distance never changes, so the progress reward is always 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if self.levels == "generated":
            level_seed = options.get("level_seed")
            if level_seed is None:
                level_seed = int(self.np_random.integers(*self.seed_range))
            # Rounded to float32 here, so the replay text gives the C++ side exactly the same value
            difficulty = np.float32(self.difficulty)
            _dll.ow_load_generated(self._world, level_seed, float(difficulty))
            self.level = f"generated {level_seed} {difficulty!s}"   # !s gives "0.3", the shortest exact float32 text
        else:
            _dll.ow_load_test_level(self._world)
            self.level = "test"
        self._prev_distance = self._distance()
        self._actions = []
        return self._observe(), {"level": self.level}

    def step(self, action):
        action = int(action)
        self._actions.append(action)
        result = _dll.ow_step(self._world, action)

        distance = self._distance()
        reward = PROGRESS_REWARD * (self._prev_distance - distance)
        self._prev_distance = distance

        terminated = result != _RUNNING
        truncated = not terminated and _dll.ow_step_count(self._world) >= self.max_steps
        info = {}
        if result == _CRASHED:
            reward += CRASH_REWARD
        elif result == _REACHED_GOAL:
            reward += GOAL_REWARD

        if terminated or truncated:
            info["result"] = {_CRASHED: "crash", _REACHED_GOAL: "goal"}.get(result, "timeout")
            info["is_success"] = result == _REACHED_GOAL
            info["level"] = self.level
            self.last_episode = (self.level, list(self._actions))

        return self._observe(), float(reward), terminated, truncated, info

    def save_replay(self, path):
        """Saves the most recently finished episode. Watch it with: outerworld.exe --replay <path>"""
        if self.last_episode is None:
            raise RuntimeError("no finished episode to save yet")
        level, actions = self.last_episode
        with open(path, "w") as f:
            f.write(f"level {level}\n")
            f.write(" ".join(map(str, actions)) + "\n")

    def close(self):
        if getattr(self, "_world", None):
            _dll.ow_destroy(self._world)
            self._world = None

    def __del__(self):
        self.close()
