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
PROGRESS_REWARD = 1.0        # per unit of distance closer to the goal
MAX_STEPS = 1200             # 20 seconds at 60 steps/s, then the episode is truncated

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
    dll.ow_reset.argtypes = [world_p]
    dll.ow_step.argtypes = [world_p, ctypes.c_int]
    dll.ow_step.restype = ctypes.c_int
    dll.ow_observe.argtypes = [world_p, ctypes.POINTER(ctypes.c_float)]
    dll.ow_goal_distance.argtypes = [world_p]
    dll.ow_goal_distance.restype = ctypes.c_float
    dll.ow_step_count.argtypes = [world_p]
    dll.ow_step_count.restype = ctypes.c_int
    return dll


_dll = _load_dll()


class OuterworldEnv(gym.Env):
    """One ship, one level. Actions 0-5 = {no thrust, thrust} x {no turn, left, right}.

    info["result"] at the end of an episode is "goal", "crash" or "timeout", and
    info["is_success"] is True when the goal was reached.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps=MAX_STEPS):
        super().__init__()
        self.max_steps = max_steps
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

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        _dll.ow_load_test_level(self._world)   # the level generator will be chosen here, using `seed`
        self._prev_distance = _dll.ow_goal_distance(self._world)
        self._actions = []
        return self._observe(), {}

    def step(self, action):
        action = int(action)
        self._actions.append(action)
        result = _dll.ow_step(self._world, action)

        distance = _dll.ow_goal_distance(self._world)
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
            self.last_episode = ("test", list(self._actions))

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
