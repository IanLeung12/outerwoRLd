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

    batch_p = ctypes.c_void_p
    int_p, float_p = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)
    dll.ow_batch_create.argtypes = [ctypes.c_int]
    dll.ow_batch_create.restype = batch_p
    dll.ow_batch_destroy.argtypes = [batch_p]
    dll.ow_batch_load_test_level.argtypes = [batch_p, ctypes.c_int]
    dll.ow_batch_load_generated.argtypes = [batch_p, ctypes.c_int, ctypes.c_uint, ctypes.c_float]
    dll.ow_batch_step.argtypes = [batch_p, int_p, ctypes.c_int, int_p, int_p, float_p, float_p]
    dll.ow_batch_observe_one.argtypes = [batch_p, ctypes.c_int, float_p]
    dll.ow_batch_distance_one.argtypes = [batch_p, ctypes.c_int, ctypes.c_int]
    dll.ow_batch_distance_one.restype = ctypes.c_float
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


_SHAPING_CODES = {"straight": 0, "path": 1, "none": 2}
_RESULT_NAMES = {_CRASHED: "crash", _REACHED_GOAL: "goal", _RUNNING: "timeout"}


class OuterworldVecEnv:
    """num_envs worlds stepped together by a single call into C++.

    Takes the same options as OuterworldEnv. Works on arrays instead of single values:

        obs = env.reset()                        # (num_envs, obs_size)
        obs, rewards, terminated, truncated, info = env.step(actions)   # actions: (num_envs,) ints

    Episodes restart automatically: when world i finishes, it is given a new level straight away,
    and obs[i] is already the first observation of that new episode. The observation the
    finished episode actually ended on is info["final_obs"][i] - use that one for bootstrapping
    a truncated episode's value.

    info["episodes"] lists the episodes that finished this step, each a dict with
    index, result ("goal", "crash" or "timeout"), is_success, level, length and return.
    """

    def __init__(self, num_envs, max_steps=MAX_STEPS, levels="test", difficulty=0.5,
                 seed_range=TRAIN_SEEDS, shaping="straight", seed=None):
        if levels not in ("test", "generated"):
            raise ValueError(f'levels must be "test" or "generated", not {levels!r}')
        if shaping not in _SHAPING_CODES:
            raise ValueError(f'shaping must be "straight", "path" or "none", not {shaping!r}')
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.levels = levels
        self.difficulty = difficulty
        self.seed_range = seed_range
        self.shaping = shaping
        self.obs_size = _dll.ow_obs_size()
        self.num_actions = _dll.ow_num_actions()
        self.rng = np.random.default_rng(seed)

        self._batch = _dll.ow_batch_create(num_envs)
        # Buffers the C++ side writes into directly
        self._obs = np.zeros((num_envs, self.obs_size), dtype=np.float32)
        self._actions = np.zeros(num_envs, dtype=np.int32)
        self._results = np.zeros(num_envs, dtype=np.int32)
        self._step_counts = np.zeros(num_envs, dtype=np.int32)
        self._distances = np.zeros(num_envs, dtype=np.float32)
        int_p, float_p = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)
        self._obs_ptr = self._obs.ctypes.data_as(float_p)
        self._actions_ptr = self._actions.ctypes.data_as(int_p)
        self._results_ptr = self._results.ctypes.data_as(int_p)
        self._step_counts_ptr = self._step_counts.ctypes.data_as(int_p)
        self._distances_ptr = self._distances.ctypes.data_as(float_p)

        self._prev_distances = np.zeros(num_envs, dtype=np.float32)
        self._returns = np.zeros(num_envs, dtype=np.float64)
        self.level_names = ["test"] * num_envs
        self._episode_actions = [[] for _ in range(num_envs)]
        self.last_episode = None    # (level, actions) of the most recently finished episode, for replays

    def _start_episode(self, i, level_seed=None):
        """Loads world i a new level and writes its first observation into the obs buffer."""
        if self.levels == "generated":
            if level_seed is None:
                level_seed = int(self.rng.integers(*self.seed_range))
            difficulty = np.float32(self.difficulty)
            _dll.ow_batch_load_generated(self._batch, i, level_seed, float(difficulty))
            self.level_names[i] = f"generated {level_seed} {difficulty!s}"
        else:
            _dll.ow_batch_load_test_level(self._batch, i)
            self.level_names[i] = "test"
        row = self._obs[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        _dll.ow_batch_observe_one(self._batch, i, row)
        self._prev_distances[i] = _dll.ow_batch_distance_one(self._batch, i, _SHAPING_CODES[self.shaping])
        self._returns[i] = 0.0
        self._episode_actions[i] = []

    def reset(self, seed=None, level_seeds=None):
        """Starts a new episode in every world. level_seeds (one per world) picks exact levels."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        for i in range(self.num_envs):
            self._start_episode(i, None if level_seeds is None else level_seeds[i])
        return self._obs.copy()

    def step(self, actions):
        self._actions[:] = actions
        _dll.ow_batch_step(self._batch, self._actions_ptr, _SHAPING_CODES[self.shaping], self._results_ptr,
                           self._step_counts_ptr, self._distances_ptr, self._obs_ptr)

        rewards = PROGRESS_REWARD * (self._prev_distances - self._distances)
        rewards[self._results == _CRASHED] += CRASH_REWARD
        rewards[self._results == _REACHED_GOAL] += GOAL_REWARD
        self._prev_distances[:] = self._distances
        self._returns += rewards
        for i in range(self.num_envs):
            self._episode_actions[i].append(int(self._actions[i]))

        terminated = self._results != _RUNNING
        truncated = ~terminated & (self._step_counts >= self.max_steps)
        info = {"final_obs": self._obs.copy(), "episodes": []}

        for i in np.flatnonzero(terminated | truncated):
            result = int(self._results[i])
            info["episodes"].append({
                "index": int(i),
                "result": _RESULT_NAMES[result],
                "is_success": result == _REACHED_GOAL,
                "level": self.level_names[i],
                "length": int(self._step_counts[i]),
                "return": float(self._returns[i]),
            })
            self.last_episode = (self.level_names[i], self._episode_actions[i])
            self._start_episode(i)

        return self._obs.copy(), rewards.astype(np.float32), terminated, truncated, info

    def save_replay(self, path):
        """Saves the most recently finished episode. Watch it with: outerworld.exe --replay <path>"""
        if self.last_episode is None:
            raise RuntimeError("no finished episode to save yet")
        level, actions = self.last_episode
        with open(path, "w") as f:
            f.write(f"level {level}\n")
            f.write(" ".join(map(str, actions)) + "\n")

    def close(self):
        if getattr(self, "_batch", None):
            _dll.ow_batch_destroy(self._batch)
            self._batch = None

    def __del__(self):
        self.close()
