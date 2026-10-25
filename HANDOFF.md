# Outerworld: session handoff

Written 2026-10-25. Read this first; it replaces the earlier conversation.

## The user and how to work with them

- Started this project to **learn C++, OpenGL and then reinforcement learning** from scratch. Experienced enough now to write PPO themselves; new to C++ and ML before this project.
- Original rule: *"only write code I tell you to write."* Over time they handed over **infrastructure** (C++ simulation, C API, Python env, generators, tooling), and asked me to do some bigger edits directly. For **RL algorithm code** they want to *understand*: explain ideas, give scaffolding or stubs when asked, review their code and point out bugs (with the reason), rather than silently writing it.
- When I make changes, they want **every change explained** afterwards.
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
- `capi.h/.cpp`: single-world `ow_*` functions and **batched `ow_batch_*`** functions (one call steps N worlds and writes results, step counts, shaping distances and observations into caller arrays).

**Game / viewer (`outerworld.exe`): `main.cpp`, `Shader`, `Mesh`, GLFW + GLAD + GLM.**
- Keyboard play of the test level: W thrust, A/D turn, Esc quit, **N/P = next/previous generated level**.
- `--seed N [--difficulty D]` plays a generated level. `--replay file` replays an episode on a loop.
- Replay file format: line 1 `level test` or `level generated <seed> <difficulty>`, line 2 the action list.

**Observation (54 floats, all in the ship's frame: x = right, y = forward):** velocity (2), sin/cos angle (2), goal relative position (2) and half size (2), distances to left/right/bottom/top walls (4, range 0–1), gravity pull in units of thrust (2), then 8 nearest obstacles sorted by surface distance × (rel x, rel y, radius/0.25, density/PLANET_DENSITY, present flag). Layout is documented in `World.h`.

**Python, `python/outerworld_env.py`:**
- `OuterworldEnv` (Gymnasium env, single world) and **`OuterworldVecEnv(num_envs, …)`** (batched, same options, same-step auto-reset; `info["final_obs"]` holds the real last observation of finished episodes; `info["episodes"]` lists finished episodes with result/is_success/level/length/return). The batched env was tested to be identical to the single env.
- Options: `levels="test"|"generated"`, `difficulty` (changeable between episodes), `seed_range` (`TRAIN_SEEDS = (0, 1e6)`, `EVAL_SEEDS = (1e9, 1e9+1e6)`), `shaping="straight"|"path"|"none"`, `max_steps=1200`.
- Reward: +10 goal, −5 crash, `+1 × (previous distance − current distance)` progress shaping, timeout = truncated.
- The DLL path is `$OUTERWORLD_DLL` or `<repo>/build/outerworld_py.dll`.

**Training, `python/outerworldrl.py` (the user's own PPO, written over the sessions):**
- Batched collection: 32 worlds × 128 steps = 4096 steps per iteration; timeouts bootstrapped by adding `γ·V(final_obs)` to the reward; vectorized GAE (γ 0.99, λ 0.95) with a done mask.
- PPO: clip 0.2, 10 epochs, minibatch 64, entropy 0.01, grad clip 0.5, KL early stop 0.02 (checked per minibatch), separate value net trained in the same minibatch loop, Adam 3e-4 (no LR annealing, because of the curriculum).
- Running observation normalization (`RunningMeanStd`, statistics frozen during collection, updated once per iteration, never from eval); orthogonal init (√2 hidden, 0.01 policy head, 1.0 value head).
- Curriculum: +0.05 difficulty when the 10-iteration average success > 80%. Eval every 10 iterations on 50 fixed `EVAL_SEEDS` levels with argmax. TensorBoard logging, `latest.pt` checkpoint every 50 iterations (in the run's log dir, includes the normalization statistics), replay of the latest episode in `runs/latest.txt`.
- `python/learning/`: the user's earlier exercises (`fit_sine.py`, `reinforce_cartpole.py`). They built REINFORCE → baseline → GAE → PPO on CartPole step by step.

## Results so far

- Random agent success by difficulty: 0.0 → 29.5%, 0.25 → 6%, 0.5 → 1%, 1.0 → 0.5%.
- PPO on the fixed test level: 100% by iteration 20 (memorized, value loss → 0).
- **Generated levels at difficulty 0 (single-env version): eval success on unseen levels tracked training success, reaching 80–95%.** It generalizes rather than memorizes, which is the project's key question answered for easy levels. A dip around iteration 100 was not fully explained.
- Batched version, quick 12-iteration check: eval 10% → 46% by iteration 10. Collection now takes about 0.25 s per 4096 steps; **the PPO update (~2.5 s per iteration) is now the bottleneck**.

## State at handoff

- **Uncommitted:** `src/capi.h`, `src/capi.cpp` (batched API), `python/outerworld_env.py` (`OuterworldVecEnv`), `python/outerworldrl.py` (batched training loop), and this file.
- **`build/outerworld_py.dll` predates the batched API.** The user must rebuild (F7) before running `outerworldrl.py`, otherwise loading fails on `ow_batch_create`.
- The previous curriculum run (single-env version) may still have been running.

## Suggested next steps (in roughly this order)

1. Rebuild, then run the batched curriculum training and check the `curriculum/difficulty`, `eval/success`, and `train/crash` vs `train/timeout` graphs.
2. **Speed up the update:** try `minibatch_size = 256` (about 4× fewer updates), comparing eval curves; then test CUDA for the update phase.
3. Experiments the user wants to run, **one change at a time**: shaping straight vs path vs none at difficulty ~0.5; network size 64 vs 128 vs 256 where the agent struggles.
4. Smaller PPO refinements discussed but not done: average the KL over each epoch instead of per minibatch; a `best.pt` checkpoint on eval improvement; `eval/timeout` logging; value LR schedule.
5. Bigger ideas discussed for later: attention/Deep Sets over obstacles, behaviour cloning from the user's own play (the replay format already records actions), DQN and HER, action repeat.
