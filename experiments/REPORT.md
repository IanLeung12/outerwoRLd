# Overnight experiments: report

2026-10-25, 05:45 to ~11:30. 63 training runs (each 600 iterations unless noted), one change at a time, every
conclusion that matters repeated with a second seed. The running log with every intermediate observation is `experiments/NOTES.md`.

## TL;DR

Success on **held-out** levels (greedy, 200 levels per difficulty, the same levels for every run):

| difficulty | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | blocker levels* |
|---|---|---|---|---|---|---|---|
| your setup before (`r1_baseline`) | 67% | 41% | 23% | 8% | 4% | 3% | 1% |
| repeat 8 + Deep Sets, 32 worlds (`r10_ds_rep8`, 2 seeds) | 90-96% | 90-94% | 86-90% | 82-85% | 76% | 68-73% | 96-98% |
| **final setup, 64 worlds (`r12_ds_rep8_envs64`, 3 seeds)** | 96-98% | 94-97% | 94% | 91-92% | 89-91% | **83-90%** | 96-99% |
| your ported `outerworldrl.py` (32 worlds, iteration 450) | 96% | 94% | 94% | 88% | 80% | 72% | 98% |
| your final `outerworldrl.py` (64 worlds, iteration 450) | 96% | 95% | 94% | 89% | 84% | 78% | 98% |
| long run, 32 worlds, 4000 its (`r11_ds_rep8_long`, ~2 h) | 94% | 94% | 93% | 93% | 94% | **93%** | 97% |
| long run, 32 worlds, 3000 its, seed 1 (`r14_ds_rep8_long_seed1`) | 99% | 95% | 93% | 97% | 90% | 91% | 98% |
| **long run, 64 worlds, 2000 its (`r16_ds_rep8_envs64_long`, ~1h45): the best agent** | **98%** | **96%** | **96%** | **95%** | **94%** | **93%** | **99%** |
| long run, 64 worlds, 1500 its, seed 1 (`r16_ds_rep8_envs64_long_seed1`) | 98% | 95% | 95% | 93% | 94% | 88% | 98% |
| best agent + 1000 more its at difficulty 1.0 (`r24_best_continued`) | 99% | 98% | 95% | 95% | 94% | 91% | 99% |
| **your `outerworldrl.py` as it is now, 2000 its (`r25_user_script_long`, ~1h45)** | 96% | **100%** | **98%** | **96%** | **96%** | **96%** | **100%** |
| r16 setup + learning-rate annealing, 2000 its (`r26_envs64_anneal_long`) | 98% | 96% | 93% | 96% | 96% | 96% | 99% |
| the same, seed 1 (`r26_envs64_anneal_long_seed1`) | 99% | 98% | 98% | 97% | 96% | 95% | 98% |
| **ensemble of 2 annealed agents, distilled into one network (`d1_ens2_init`): the best single policy** | **98%** | **98%** | **99%** | **97%** | **98%** | **96%** | **100%** |

\* difficulty 0 levels that always have a planet across the straight line to the goal.

The 600-iteration rows take ~15-25 minutes with 32 worlds, ~40 with 64. The agent does **as well on held-out levels as on
training levels** (1000 levels each: 72% vs 69% at difficulty 1.0), so it learned skills, not levels.

Five things made the difference, in order of discovery:

1. **Action repeat 8**: each decision is held for 8 physics steps. Fixes the "can't dodge" problem (credit horizon).
2. **Blocker levels in training** (half the levels have a planet across the straight line), plus starting the
   curriculum at 0. Teaches "dodge when something is in the way" without unlearning "fly straight when it isn't".
3. **No observation normalization.** RunningMeanStd had a real bug for this observation (details below).
4. **Deep Sets** for the planets: one shared network per planet, max-pooled. Generalizes to more planets.
5. **64 parallel worlds** instead of 32: more different levels per update. Better even at equal training time.

(16 ray-cast distances, "lidar", helped a lot before Deep Sets; with Deep Sets they're optional. And with all of
the above, the best agent even handles levels with 9-12 planets, more than it ever trained on: 83% / 69%.)

Later in the morning: **learning-rate annealing** adds 1-2 points, and averaging two agents (an **ensemble**) and then
**distilling** it into one network gives the best single policy: 96% at difficulty 1.0 on 1000 held-out levels. See
"After the report" below.

![progress](progress.png)

All of these are now in `python/outerworldrl.py` (your old version is in
`experiments/backup/outerworldrl_before_overnight.py`). **Press F7 before running it**: it needs the new DLL.

## What was wrong, and why each fix works

### 1. The agent couldn't learn to fly around a planet: credit horizon

What you saw yesterday: the agent flies straight at the goal and crashes whenever a planet is in the way.

GAE with gamma 0.99 and lambda 0.95 weights a reward k steps in the future by (0.99 x 0.95)^k = 0.94^k. That
halves every ~11 steps, so the advantage of an action mostly reflects the next ~17 steps = 0.3 seconds. But
the ship has momentum and no brakes, so a dodge has to start 0.5-1 s before the planet. The -5 for crashing
gets blamed on the last 0.3 s, when nothing can be done any more; the decision that mattered (turning
earlier) gets almost no signal.

**Action repeat** applies each chosen action for k physics steps. Nothing changes in GAE, but now 17 decisions
cover 17 x k frames: 1.1 s with k = 4, 2.3 s with k = 8. Fewer decisions per episode also makes each one matter
more, so exploration is more "purposeful": a random decision changes the trajectory for 8 frames, not 1.

Evidence:
- On blocker-only levels, repeat 4 reached 96% training success by iteration ~180; repeat 1 needed 600 iterations
  to reach 84%.
- **Control** (`r2_control_steps512`): repeat 1 but 4x more frames per iteration (so the same data as repeat 4)
  got 3% on blocker levels vs 42% for repeat 4. The gain is the decision horizon, not more data.
- lambda 0.98 (the other way to lengthen the horizon) alone did nothing measurable.
- Repeat 4 < **8** > 16. At 16 (0.27 s per decision) steering gets too coarse (blocker levels drop to 65%).

### 2. What it trains on: blocker levels + curriculum from 0

At difficulty 0-0.2 most levels have a clear straight path, so "fly straight at the goal" already scores
well, and the curriculum gate (80%) is right at that plateau. I added a generator option (`blockerChance`) that
puts one planet across the spawn-goal line.

- **Only** blocker levels (`r1_blockonly*`): the agent learns to swerve (98% on blocker levels) but then
  swerves on clear levels too: 14-43% at difficulty 0. It learned "always swerve".
- **Half** blocker levels (`--blocker-chance 0.5`): learns both. With repeat 4: 0.4 difficulty 30% -> 59%.
- 25% blockers: the same as 50% within noise (two seeds each).
- No curriculum (uniform difficulty 0-1 from the start): much worse (41% at difficulty 0). Curriculum needed.
- A 70% promotion gate (instead of 80%) moved the curriculum faster but the agent wasn't better on hard levels.
  **The curriculum level is not a good progress measure by itself**; the fixed-difficulty evals are.
- With the final setup (Deep Sets + repeat 8), removing the blockers still costs ~0.15 on hard levels
  (two seeds: 0.60/0.53 at 0.8/1.0 vs 0.76/0.71; blocker levels 72-88% vs 96-98%), so they stay.
- With the final setup, **no curriculum at all** (each episode's difficulty uniform in 0..1) is only a little
  worse: two-seed mean 0.69/0.66 at 0.8/1.0 vs 0.76/0.71. With the early setup it was a disaster (41% even at
  difficulty 0). Once the agent can perceive planets and assign credit properly, the curriculum stops being
  essential; it still helps a bit, so it stays.

### 3. Bug: observation normalization breaks on features that never varied in training

`analyze.py` (new tool, see below) showed a cliff on a checkpoint trained up to difficulty 0.5: at difficulty 0.8,
levels with 6-8 planets had **1%** success (97% crash) vs 55% with 4-5 planets.

Cause: at difficulty <= 0.5 a level has at most round(2 + 6 x 0.5) = 5 planets, so obstacle slots 6-8 were
**always zero** during training. RunningMeanStd gives them variance 0, and normalizing divides by
sqrt(0 + 1e-8) = 1e-4: a real planet appearing in slot 6 becomes (0.5 - 0) / 1e-4, clipped to +-10. The network
gets huge garbage inputs it has never seen.

Proof without retraining: copying slot 5's statistics into slots 6-8 of that checkpoint raised the 6+ planet
levels from 1% to 28%.

The same bug in reverse caused the agent to "forget" difficulty 0 (57% there vs 89% at 0.2): 151 of 170 crashes
were into **walls** on levels with 0-1 planets. Slot 1 was filled 98.7% of the time in training (std 0.11), so an
**empty** slot 1 normalizes to (0 - 0.987) / 0.11 = -9.

In a curriculum this bites every time a promotion unlocks a new planet count. Your observation is already
scaled to about -1..1 in C++, so the fix is simply not to normalize: difficulty 0 went 67% -> 93-95%, 6+ planets
at 0.8 went 1% -> 35% (same everything else). A std floor of 0.1 would not have fixed the slot-1 case.

General lesson: running normalization assumes each feature's training distribution covers what it will see
later. Features that are constant in training (padding slots, flags) break that.

### 4. Deep Sets: the network handles planets one at a time

The MLP gives each obstacle slot its own weights. Slot 7's weights only learn from levels with 7+ planets,
which are rare (and absent early in the curriculum), so they stay nearly untrained. And it has to learn the
same "is this planet dangerous" rule 8 separate times.

Deep Sets (`PlanetSetNet` in your script, `experiments/models.py`): each slot's 5 numbers go through the same
small network phi (5 -> 32 -> 32), the outputs are max-pooled over the planets that are present (empty slots
are zeroed first; ReLU outputs are >= 0, so zeros never win the max), and the 32-number summary joins the rest of
the observation in the usual 64x64 MLP. Order and count don't matter, and every planet trains the same weights.

Evidence (2 seeds each, repeat 4): difficulty 0.8 / 1.0 went 0.52/0.34 and 0.42/0.23 (MLP) -> 0.71/0.61 and
0.65/0.53 (Deep Sets). At 1.0, levels with 6-8 planets: 19% -> 49%. With repeat 8 the two gains add up
(0.76/0.73 and 0.76/0.68). Variations that made no difference: max+mean pooling, a bigger network
(128 hidden, 64 embedding).

### 5. Rays ("lidar")

16 distances from the ship to the first thing it would crash into (planet or wall, both grown by the ship's
radius), evenly spaced clockwise from straight ahead, / 1.5, capped at 1. They make "something is in front of me"
directly visible instead of something the network computes from (x, y, radius) triples.
- Rays alone (repeat 1): small gain. With repeat 4 (and the MLP): big gain (difficulty 0.4 30% -> 65%).
- With Deep Sets + repeat 8, removing them: two-seed means at 0.8/1.0 went 0.76/0.71 -> 0.71/0.65. Within noise.
  Deep Sets already sees the planets well; the rays are optional. I kept them in your script (cheap).

### 6. More worlds per iteration

64 worlds x 128 decisions = 8192 per iteration instead of 4096. Per iteration it's twice the data and about
twice the time, so the fair comparison is at equal **wall-clock**: at ~40 minutes the 32-world long run was at
83% (fixed eval, difficulty 0.8) vs 89-91% for 64 worlds, and 64 worlds reached curriculum 1.0 within 600
iterations. Each update averages over twice as many different levels, so the gradient is less noisy and less
specific to a few levels. Two seeds. `NUM_ENVS = 64` in your script now.

## Things that made no real difference

| change | result |
|---|---|
| lambda 0.98 | same as 0.95 |
| KL early stop per epoch on the whole batch (instead of per minibatch) | same. With minibatch 256 the early stop almost never fires anyway (9.9 of 10 epochs on average) |
| 128 hidden units instead of 64 | same (network size is not the bottleneck) |
| straight-line shaping instead of path (distance field) | same once the other fixes are in. Answers your "straight vs path" experiment |
| entropy 0.003 instead of 0.01 | slightly worse on hard levels |
| sampling each episode's difficulty in [0, curriculum level] | no clear gain (raises train success, so the curriculum level climbs faster, but not real skill) |
| learning rate 1e-3 | same |
| no shaping at all (only +10 goal / -5 crash) | almost as good: 0.70/0.56 at 0.8/1.0 vs 0.76/0.71 with path shaping. Completes your straight / path / none question: with repeat 8 + curriculum the sparse reward is nearly enough |
| 4 PPO epochs instead of 10 | worse per iteration, not better per minute |
| max+mean pooling, bigger Deep Sets (128 / 64) | same |
| 25% blocker levels instead of 50% | same |
| lambda 0.90 | same |
| timeouts not bootstrapped (`--timeout-terminal`) | no better, more crashes |
| sampling difficulty in [0, level] on a long run | better on easy levels, worse at 1.0 (83% vs 91-93%) |
| action repeat 12 with Deep Sets | same as 8 or slightly worse (0.71/0.65 at 0.8/1.0). 8 stays |
| 128 worlds for 300 its (same frames and time as 64 worlds x 600) | worse (0.73 at 1.0 vs 0.83-0.90): half as many updates. 64 is the sweet spot |

Seed-to-seed spread is about +-0.1 at difficulty 0.8-1.0, so I only trusted differences bigger than that or seen
on two seeds.

### 7. Zero-shot: levels with more planets than it has ever seen

A Deep Sets network doesn't care how many obstacle slots the observation has. So I built a scratch copy of the
project with `OBS_MAX_OBSTACLES = 12` (your source is untouched) and let the best agent, trained on at most 8
planets, play levels with up to 12, without any retraining. 1000 levels at difficulty 1.0: **85% overall;
9-10 planets 83%, 11-12 planets 69%** (4-8 planets: 91-96%). Letting it see only the nearest 8 planets gives
almost the same numbers (84%), so far-away planets rarely matter and the drop on very busy levels is them
being genuinely harder. The old MLP couldn't even accept a 12-slot observation.

## Remaining failures, and a deadlock

The 4000-iteration agent (`r11_ds_rep8_long`) at difficulty 1.0 on 1000 held-out levels: 90% success. The 95
failures: 32 planet crashes, 28 wall crashes and **35 timeouts**. Planet count, turn needed at spawn, distance and
goal size barely matter any more (88-98% in every bucket).

The timeouts are a **greedy-policy deadlock**: the ship never moves at all. Either it picks action 0 (no thrust,
no turn) forever, or it alternates left / right forever. Spawn is outside all gravity, so if argmax says "do
nothing" the observation never changes and argmax says it again; left-then-right returns to the same state. With
*sampled* actions the same policy solves those levels most of the time (55 of 70 tries).

Why the policy is happy to wait: nothing makes waiting cost anything. A timeout is bootstrapped with V(final
state), which for a ship still at spawn is just V(spawn). So when it is unsure, "do nothing" is never punished.
I tried two fixes: a small per-decision time penalty (`--step-penalty`) and not bootstrapping timeouts
(`--timeout-terminal`). Neither is worth it:
- Not bootstrapping timeouts: no better, more crashes.
- A penalty of 0.03 per decision on a 64-world long run: timeouts at difficulty 1.0 dropped from 14 to 6 per 1000
  levels, but crashes rose from 64 to 112 (it hurries), so overall 88% vs 92%.

With 64 worlds and a long run the deadlocks are down to ~1% of levels anyway, and the cheap fix is at evaluation
time: **greedy, but sample whenever the ship hasn't moved for 8 decisions (~1 s)** (`analyze.py --unstick 8`).
On the best agent, 1000 levels each, it removes every timeout without adding crashes: difficulty 0.5 96% -> 98%,
difficulty 1.0 92% -> **94%**. (Your `evaluate()` still uses plain argmax; easy to add if you like it.)
Training the best agent 1000 iterations more (`r24_best_continued`) cut crashes at 1.0 (64 -> 47 per 1000) but
added deadlocks (14 -> 54), so greedy is 90%, and greedy + unstick is **95%**, the best result of the night.

Last run of the night: **your own `python/outerworldrl.py`, unchanged except 2000 iterations** (`r25_user_script_long`,
checkpoint `runs/exp/r25_user_script_long/latest.pt`). On 1000 held-out levels: difficulty 0.5 **97%**, difficulty
1.0 **94% greedy with zero timeouts** (r16: 92% greedy with 14 timeouts, r24: 90% with 54). It never deadlocked,
so unstick changes nothing for it; with unstick the three are tied at 94-95%. It's the same recipe as r16 (the gap
is within seed noise), but it means the script in your repo, as is, trains the best agent of the night.

Replays (after F7): `runs/exp/r11_ds_rep8_long/replays/` (12 successes at difficulty 1.0) and
`runs/exp/r11_ds_rep8_long/replays_failures/` (all 15 failures among the first 200 levels, 7 of them deadlocks):
`build/outerworld.exe --replay runs/exp/r11_ds_rep8_long/replays_failures/level17_timeout.txt`.

## After the report (late morning): annealing, failure overlap, ensembles

All numbers here are on **1000** held-out levels at difficulty 1.0 (noise about +-0.6 points).

**Learning-rate annealing** (`train.py --anneal-lr`: lr goes linearly from 3e-4 to 0 over the run; a standard
PPO trick we hadn't tried). Two seeds of the r16 setup, 2000 iterations:

| agent | greedy | greedy + unstick | timeouts (greedy) |
|---|---|---|---|
| r16 (no annealing) | 92.2% | 93.6% | 14 |
| r25 (your script, no annealing) | 93.8% | 93.8% | 0 |
| r26 annealed, seed 0 | 94.3% | 94.5% | 2 |
| r26 annealed, seed 1 | 94.8% | **95.4%** | 8 |

A small, consistent gain (both seeds beat r16), mostly fewer deadlocks. Worth adding to your script: set
`g["lr"] = 3e-4 * (1 - iteration / num_iterations)` on both optimizers at the top of each iteration.

**Are the failures hard levels, or agent mistakes?** I saved each of the four agents' result on every level
(`analyze.py --dump`) and compared:
- Each agent fails 46-64 of the 1000 levels, but the failures barely overlap: any two agents share only 18-28.
  **Only 10 levels beat all four agents**, so "some agent succeeds" would be 99%.
- The best agent's own failures are mostly solvable *by it*: sampling actions instead of argmax (4 tries) solves
  36 of its 46 failed levels at least once. Only 6 of the 1000 levels were never solved by anything.
So the last ~5% isn't impossible levels. It's idiosyncratic mistakes, where each agent zigs wrong at a different moment.

**That's exactly what ensembles fix.** `analyze.py --ensemble a.pt,b.pt` averages the agents' action probabilities
and takes the argmax:

| policy | difficulty 0.5 | difficulty 1.0 |
|---|---|---|
| best single agent (r26 seed 1), greedy | 97.3% | 94.8% |
| ensemble of the 2 annealed agents | - | **96.9%** (30 crashes, 1 timeout) |
| ensemble of 4 (r26 x2, r25, r16) | **98.7%** | 96.5% |

Two agents are as good as four. It costs 2x the compute per decision at play time (still trivial here).

**Distilling the ensemble into one network** (`experiments/distill.py`). Each round, the student plays training levels
(sampling its own actions, so it visits the states *it* gets into, including its mistakes). The two annealed agents
label every visited state with their mean action probabilities, and the student is trained to match them
(cross-entropy to the soft targets). This is DAgger with an ensemble as the expert. About 2.3 s per round.

| policy (1000 held-out levels) | difficulty 0.0 | 0.5 | 1.0 | blocker levels |
|---|---|---|---|---|
| best single PPO agent (r26 seed 1) | - | 97.3% | 94.8% | - |
| **student, initialized from r26 seed 1, 300 rounds (~12 min) (`d1_ens2_init`)** | 99.2% | **98.7%** | **96.2%** | 99.5% |
| student from random weights, 600 rounds (~25 min) (`d2_ens2_fresh`) | 99.4% | 96.8% | 95.4% | 99.3% |
| control: r26 seed 1 distilled into itself, 300 rounds (`d3_self_control`) | - | 97.3% | 94.7% | - |

- The student matches the ensemble in **one network**: `runs/exp/d1_ens2_init/final.pt` is the best single policy of
  the night. Load it like any other checkpoint (same format and observation setup).
- The control shows the gain is the ensemble's knowledge, not just more training at a low learning rate.
- From random weights, distillation reaches the best PPO agent's level in ~25 minutes, versus 1h45 of PPO. That's
  because copying an expert is supervised learning with a dense signal (the full action distribution on every state)
  instead of trial and error. Of course it needed the experts first.


**C++ (rebuild with F7):**
- `LevelGen.h/.cpp`: `GenParams::blockerChance`. With that chance, a planet (radius >= 0.12) is placed across the
  spawn-goal line before the other obstacles. When it's 0 no extra random numbers are drawn, so **every existing
  seed gives exactly the same level as before** (checked: 3000 levels, old vs new DLL, 0 differences).
- `capi.h/.cpp`: `ow_batch_load_generated_ex` (with blocker chance), `ow_batch_step_repeat` (action repeat;
  `ow_batch_step` now calls it with repeat 1), `ow_batch_rays` (lidar).
- `main.cpp` (viewer): `--blocker <chance>`, and replay lines may have a 4th number (the blocker chance).

**Python:**
- `outerworld_env.py`, `OuterworldVecEnv`: new options `action_repeat`, `blocker_chance`, `rays`, `ray_range`.
  Defaults keep the old behaviour. With action repeat, replays still record every physics step (the viewer
  needed no change; checked that replays re-simulate to the same result). `obs_size` includes the rays.
- `outerworldrl.py`: the best setup (see the top of the file: NUM_ENVS 64, ACTION_REPEAT, RAYS, BLOCKER_CHANCE,
  start difficulty 0; PlanetSetNet; normalization removed with a comment; checkpoints save "args" for the tools
  below). Everything else (PPO, GAE, curriculum, eval, logging) is your code unchanged. I trained it as-is for 500 iterations (~35 min): at iteration 450 it scores
  0.96/0.95/0.94/0.89/0.84/0.78 (difficulty 0..1) and 98% on blocker levels; checkpoint in `runs/exp/r20_user_script_64/latest.pt`.

**New, in `experiments/`:**
- `train.py`: your PPO with every setting as a flag, fixed-difficulty evals, `results.json`, `best.pt`.
- `summarize.py`: one line per run. `analyze.py`: success by level feature, wall vs planet crashes,
  `--train-seeds` for the generalization check, `--sampled`. `replay.py`: save replays of a checkpoint.
  `models.py`: networks. `plot_progress.py`: the chart above. `NOTES.md`: the full log.
- Runs are in `runs/exp/<name>/` (TensorBoard: `python -m tensorboard.main --logdir runs/exp`).

## Suggested next steps

**To pick up where this left off:**
1. Press **F7** (rebuilds `build/` with the new C++: blockers, action repeat, rays). Then
   `python python/outerworldrl.py` works as before (~35 min for 500 iterations with 64 worlds).
2. Watch the best agent: `build/outerworld.exe --replay runs/exp/r16_ds_rep8_envs64_long/replays/level0_goal.txt`,
   and its predecessor's failures in `runs/exp/r11_ds_rep8_long/replays_failures/`.
3. TensorBoard for everything: `python -m tensorboard.main --logdir runs/exp`. Each run's settings are in its
   `args.json`, its final numbers in `results.json`.
4. Nothing is committed. Your old training script is in `experiments/backup/`. A sensible split for commits:
   C++ + env (blockers, repeat, rays), the training script changes, and `experiments/`.

**Ideas worth trying next (my opinion, roughly by value):**
- **Re-implement the four main changes yourself** from the diff of `outerworldrl.py`: you'll understand them far
  better. In particular, rework the GAE horizon arithmetic with and without action repeat, and try to
  reproduce the normalization bug on purpose (train at difficulty 0.3, evaluate at 0.8 with and without
  RunningMeanStd).
- **Train with 12 slots** (section 7 only tested it). Raising `OBS_MAX_OBSTACLES` changes the levels the generator
  makes at higher difficulty (its maximum planet count uses that constant), so every seed above difficulty 0 would
  change; I'd separate the generator's maximum from the observation's slot count first.
- **Attention over planets** (a small transformer layer instead of max pooling): lets planets "see" each other,
  e.g. two planets with a gap between them.
- **Generalizing over physics**: randomize thrust / drag / gravity per level and see whether one agent handles
  them all (add them to the observation, or not, and compare).
- **Behaviour cloning** from your own play (replays record actions; with action repeat, take every 8th action),
  then fine-tune with PPO. Compare learning speed with and without the head start.
- **DQN** on the same environment as a learning exercise: discrete actions, so it applies directly; action repeat
  and Deep Sets carry over.
