# Outerworld: session handoff

Written 2026-10-25, updated after the overnight experiments (2026-10-25 ~11:30). Read this first; it replaces the earlier
conversation. **Then read `experiments/REPORT.md`** (what changed overnight and why).

## The user and how to work with them

- Started this project to **learn C++, OpenGL and then reinforcement learning** from scratch. Experienced enough now to write PPO themselves; new to C++ and ML before this project.
- Original rule: *"only write code I tell you to write."* Over time they handed over **infrastructure** (C++ simulation, C API, Python env, generators, tooling), and asked me to do some bigger edits directly. For **RL algorithm code** they want to *understand*: explain ideas, give scaffolding or stubs when asked, review their code and point out bugs (with the reason), rather than silently writing it.
- When I make changes, they want **every change explained** afterwards.
- For the overnight run they said: "feel free to touch the rl file i dont really care just when i get back i need explanations". `outerworldrl.py` was rewritten to the best setup; their old version is `experiments/backup/outerworldrl_before_overnight.py`.
- Windows 11, PowerShell, VS Code with CMake Tools (F7 builds, Shift+F5 runs). Commits and pushes themselves (remote: `github.com/IanLeung12/outerworld`).

## Goal of the project

A 2D physics game: steer a ship (thrust and turn) from a spawn point to a goal, avoiding circular planets (instant crash), some with gravity. The real aim is to **train an RL agent that generalizes**: it learns general flying skills on procedurally generated levels, measured on held-out levels it never trained on.

## Toolchain facts

| Thing | Detail |
|---|---|
| Compiler | Strawberry Perl's MinGW g++ 13.2 (UCRT), CMake 3.29, Ninja. Build dir `build/`. |
| Optimization | `outerworld_core` is compiled with `-O2` (CMakeLists). No `CMAKE_BUILD_TYPE` is set otherwise. |
| Python | 3.14.6 (MSVC build), torch 2.11.0+cu128, RTX 5060 Laptop GPU (CUDA works), gymnasium 1.4, tensorboard 2.21, numpy, matplotlib. |
| Python ↔ C++ | A **plain C API** (`src/capi.h`) in `outerworld_py.dll`, loaded with ctypes. Not pybind11, because MinGW and MSVC C++ ABIs are incompatible. The DLL is statically linked (no MinGW DLLs needed). |
| TensorBoard | `python -m tensorboard.main --logdir runs` (pip's Scripts folder isn't on PATH). |
| Profiling | Use `python -m cProfile -s cumtime script.py`. The `python-profiling` backport of 3.15's sampling profiler is **broken on Windows/3.14** (ships the wrong `_remote_debugging` module); the user was advised to uninstall it. |
| **DLL lock** | While a training run is going, Windows locks `build/outerworld_py.dll`, so a rebuild into `build/` fails. I always built and tested in a **scratch build dir** instead and told the user to press F7 after their run. |

## Code layout

**C++ simulation (no OpenGL), library `outerworld_core`:**
- `Ship.h/.cpp`: position, velocity, angle; `update(dt, ShipInput, externalAccel)` (thrust, turn, gravity, drag via `pow(SHIP_DRAG, dt)`). `ShipInput` has 3 bools. `decodeAction/encodeAction`: 6 actions = `thrust*3 + turn` (turn 0 none, 1 left, 2 right).
- `World.h/.cpp`: fixed timestep `SIM_DT = 1/60`, deterministic. `step(input) → Running | Crashed | ReachedGoal`, `observe()` (54 floats), `goalDistance()` (straight line), `pathDistance()` (distance field, built lazily ~2–3 ms per level), `load(level)`, `reset()`.
- `Level.h` (`Level` struct + `makeTestLevel()`), `Obstacle.h` (circle, `density`, inverse-square gravity `density*r*(r/d)^2`, cutoff at `GRAVITY_RANGE*r`), `Goal.h` (axis-aligned rect, `halfSize`), `collision.h` (circle–circle, circle–rect).
- `LevelGen.h/.cpp`: `generateLevel(seed, GenParams)`, `paramsForDifficulty(0..1)`. Portable RNG (raw mt19937 → floats). Rejection sampling; ship-sized gaps between obstacles and from edges guarantee a route; spawn outside all gravity; limited gravity at the goal. Tested on 30k levels.
- `DistanceField.h/.cpp`: 178×100 grid, Dijkstra from goal cells, bilinear lookup.
- `constants.h`: all tuning values (world is ±16/9 × ±1; `SHIP_THRUST 2.0`, `SHIP_TURN_SPEED 3.5`, `SHIP_DRAG 0.75`, `SHIP_RADIUS 0.04`, `PLANET_DENSITY 12`, `GRAVITY_RANGE 3`, observation sizes and scales).
- `capi.h/.cpp`: single-world `ow_*` functions and **batched `ow_batch_*`** functions (one call steps N worlds and writes results, step counts, shaping distances and observations into caller arrays). Overnight additions: `ow_batch_load_generated_ex` (blocker chance), `ow_batch_step_repeat` (action repeat), `ow_batch_rays` (lidar).
- `GenParams::blockerChance` (LevelGen): chance of a planet across the spawn-goal line; 0 draws no extra random numbers, so existing seeds are unchanged. Viewer: `--blocker <chance>`, replay lines may carry a 4th number.

**Game / viewer (`outerworld.exe`): `main.cpp`, `Shader`, `Mesh`, GLFW + GLAD + GLM.**
- Keyboard play of the test level: W thrust, A/D turn, Esc quit, **N/P = next/previous generated level**.
- `--seed N [--difficulty D]` plays a generated level. `--replay file` replays an episode on a loop.
- Replay file format: line 1 `level test` or `level generated <seed> <difficulty>`, line 2 the action list.

**Observation (54 floats, all in the ship's frame: x = right, y = forward):** velocity (2), sin/cos angle (2), goal relative position (2) and half size (2), distances to left/right/bottom/top walls (4, range 0–1), gravity pull in units of thrust (2), then 8 nearest obstacles sorted by surface distance × (rel x, rel y, radius/0.25, density/PLANET_DENSITY, present flag). Layout is documented in `World.h`.

**Python, `python/outerworld_env.py`:**
- `OuterworldEnv` (Gymnasium env, single world) and **`OuterworldVecEnv(num_envs, …)`** (batched, same options, same-step auto-reset; `info["final_obs"]` holds the real last observation of finished episodes; `info["episodes"]` lists finished episodes with result/is_success/level/length/return). The batched env was tested to be identical to the single env.
- Options: `levels="test"|"generated"`, `difficulty` (changeable between episodes), `seed_range` (`TRAIN_SEEDS = (0, 1e6)`, `EVAL_SEEDS = (1e9, 1e9+1e6)`), `shaping="straight"|"path"|"none"`, `max_steps=1200`. `OuterworldVecEnv` also: `action_repeat`, `blocker_chance`, `rays`, `ray_range` (obs_size = 54 + rays).
- Reward: +10 goal, −5 crash, `+1 × (previous distance − current distance)` progress shaping, timeout = truncated.
- The DLL path is `$OUTERWORLD_DLL` or `<repo>/build/outerworld_py.dll`.

**Training, `python/outerworldrl.py`** (the user's PPO, now with the overnight best setup): 64 worlds x 128 decisions,
action repeat 8, 16 rays, blocker chance 0.5, curriculum from difficulty 0 (+0.05 when 10-iteration train success > 80%),
**no observation normalization** (RunningMeanStd broke on never-varying features), **Deep Sets** policy and value nets
(`PlanetSetNet`), PPO as before (clip 0.2, 10 epochs, minibatch 256, entropy 0.01, KL stop 0.02, gamma 0.99 and lambda 0.95
per decision). Diagnostics: approx_kl, entropy, epochs, explained variance, blocked/clear eval split.

**Experiments, `experiments/`:** `train.py` (same PPO, every setting a flag, results.json), `summarize.py`, `analyze.py`
(success by level feature, wall vs planet crashes), `replay.py`, `models.py`, `plot_progress.py`, `NOTES.md` (full log),
`REPORT.md`. Runs in `runs/exp/<name>/`.
- `python/learning/`: the user's earlier exercises (`fit_sine.py`, `reinforce_cartpole.py`).

## Results so far

- Before overnight: the agent couldn't fly around planets (credit horizon), and held-out success was 67/41/23/8/4/3% at
  difficulty 0/0.2/0.4/0.6/0.8/1.0.
- After: best agent (`runs/exp/r16_ds_rep8_envs64_long/final.pt`, 2000 its) 98/96/96/95/94/93%; 600-iteration runs of
  the final setup 96-98% ... 83-90%. `python/outerworldrl.py` as is, 2000 its (`runs/exp/r25_user_script_long/latest.pt`):
  96/100/98/96/96/96%, 94% on 1000 levels at 1.0 with no deadlocks. LR annealing (`train.py --anneal-lr`)
  adds ~1-2 points (95% at 1.0). The last ~5% failures are agent-specific (only 1% of levels beat every agent), so
  an ensemble of 2 agents (`analyze.py --ensemble`) reaches 97% at 1.0. See REPORT.md "After the report".
  Distilling that ensemble into one net (`experiments/distill.py`) gives `runs/exp/d1_ens2_init/final.pt`:
  96% at 1.0, 99% at 0.5, the best single policy. Held-out = training-level success (no memorization). Zero-shot on 9-12 planets: 83%/69%.
- Things that didn't matter: lambda, KL-stop mode, network size, shaping type (even none is close), entropy, lr,
  pooling type, rays (once Deep Sets is in). Details and numbers: `experiments/REPORT.md`.
- Known minor issue: greedy-policy deadlock (ship never moves) on ~1% of hard levels; step penalty / terminal timeouts
  didn't help overall.

## State at handoff (after overnight)

- **Uncommitted:** C++ (`LevelGen`, `capi`, `main.cpp`), `python/outerworld_env.py`, `python/outerworldrl.py`,
  `experiments/`, this file. The user commits themselves.
- **`build/` predates the overnight C++ changes: press F7 before running anything.** Overnight runs used scratch
  builds via `OUTERWORLD_DLL`.

## Suggested next steps

See the end of `experiments/REPORT.md`: re-implement the changes to understand them, train with more obstacle slots
(separate the generator's max planet count from `OBS_MAX_OBSTACLES` first), attention over planets, randomized physics,
behaviour cloning from the user's play, DQN as an exercise.
