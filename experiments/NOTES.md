# Overnight experiments: lab notebook

Started 2026-10-25 ~05:50. Autonomous: Claude runs experiments while the user sleeps.
(The final summary for the user goes in `REPORT.md`; this file is the running log.)

## Starting point / problem

- The curriculum run (starting at difficulty 0) plateaued at ~75% because the agent never learned to fly
  **around** a planet: on held-out levels at difficulty 0.05, 22/100 levels were never solved and 73% of those
  had a planet on the straight spawn->goal line (vs 5% of solved levels). It flies straight at the goal.
- Starting the curriculum at 0.2 (36% of levels blocked): after 250 iterations, blocked-level success
  was still 4% (sampled) / 8% (greedy) vs 53% / 49% on clear levels. So the skill still isn't learned.
- Hypothesis: credit horizon. GAE with gamma 0.99, lambda 0.95 looks ~17 steps (0.3 s) ahead,
  but a dodge must start ~0.5-1 s before impact.

## Infrastructure added tonight

- **Blocker levels** (`GenParams::blockerChance`, C++): with that chance, one planet (radius >= 0.12) is placed
  across the spawn->goal line before the other obstacles. Draws no random numbers when 0, so all existing seeds
  give identical levels (verified: 3000 levels, old vs new DLL, 0 differences). At blocker 1.0, 100% of levels
  are blocked. 30k levels at difficulty 0 checked for a route (the distance field finds one).
  Exposed as `ow_batch_load_generated_ex`, `OuterworldVecEnv(blocker_chance=...)`, and in the viewer:
  `--blocker <chance>` and replay lines `level generated <seed> <difficulty> <blocker>`.
- **Action repeat** (`ow_batch_step_repeat`, `OuterworldVecEnv(action_repeat=k)`): each decision is applied for
  k timesteps (stopping early if the episode ends). Verified identical to k manual steps. Replays still record
  every timestep, so the viewer needs no change.
- `experiments/train.py`: the user's PPO with every setting as a flag + fixed-difficulty evals (0.2, 0.5 on 100
  held-out levels, every 25 iterations), blocked/clear split, best.pt, results.json (final eval at
  0, .2, .4, .6, .8, 1 and blocker-only levels).
- `experiments/summarize.py`: one line per run.
- Experiments use a DLL built in a scratch dir (OUTERWORLD_DLL), so `build/` is untouched. F7 will rebuild it
  with the new features.

## Round 1 (600 iterations each, 6 in parallel, seed 0)

Defaults = the user's current settings: start difficulty 0.2, train-success curriculum (>80% over 10 its, +0.05),
path shaping, minibatch 256, lambda 0.95, gamma 0.99, ent 0.01, 64x64.

| run | change | question |
|---|---|---|
| r1_baseline | none | reference |
| r1_repeat4 | action repeat 4 | does a longer effective horizon help? |
| r1_lam98 | lambda 0.98 | same question, via GAE |
| r1_blockonly | difficulty 0 + blocker 1.0, no curriculum | can avoidance be learned at all when it's the only skill? |
| r1_blockonly_repeat4 | same + repeat 4 | |
| r1_klepoch | KL early stop on whole batch per epoch | does the noisy per-minibatch KL stop cut learning short? |

### Round 1 observations (as they came in)

- ~06:05, r1_blockonly finished (repeat 1, blocker-only levels at difficulty 0): train success climbed to 84%,
  so avoidance IS learnable with repeat 1, just slowly. But final greedy eval on normal levels: 14% at
  difficulty 0, 20% at 0.2 (78% on blocker levels). It learned "always swerve", not "swerve when needed":
  **training only on blocked levels doesn't transfer**. Need a mix.
- At ~iteration 180, r1_blockonly_repeat4 was already at 96% train success: repeat 4 learns avoidance far faster.
- r1_repeat4 (normal curriculum) leads the curriculum runs (fixed 0.2: 69% vs 46-51%), mostly on clear levels;
  its blocked-level success at 0.2 is 28% (others 12-20%).

## Infrastructure added during round 1

- **Rays** (`ow_batch_rays`, `OuterworldVecEnv(rays=N, ray_range=1.5)`, `train.py --rays N`): N "lidar"
  distances to the first planet-or-edge (grown by the ship radius, i.e. where a crash would happen), evenly
  spaced clockwise from straight ahead, /range, capped at 1. Appended after the 54 normal values. Verified
  against brute-force ray marching (max error 0.0007, the march step). Built in a second scratch dir (build2)
  because the round-1 processes lock the first DLL.
- Round 1 final greedy evals (200 held-out levels each; difficulty 0/.2/.4/.6/.8/1 | blocker-only levels at 0):
  - r1_baseline: 0.67/0.41/0.23/0.08/0.04/0.03 | 0.01. The curriculum never left 0.2 in 600 iterations.
  - r1_lam98:    0.64/0.44/0.14/0.04/0.01/0.00 | 0.02. lambda 0.98 alone: no real gain.
  - r1_klepoch:  0.67/0.51/0.20/0.09/0.01/0.01 | 0.01. Whole-batch KL per epoch: a small gain at 0.2, within noise.
  - r1_blockonly: 0.14/0.20/0.09/0.06/0.01/0.01 | 0.78 (see above).

## Round 2 (launched as round-1 slots freed up)

| run | change vs baseline | question |
|---|---|---|
| r2_rep4_mix50_d0 | repeat 4, blocker 0.5, start difficulty 0 | does a mix of blocked + clear levels teach "dodge when needed"? |
| r2_rep4_rays16 | repeat 4, 16 rays | does lidar make obstacles easier to perceive? |
| r2_control_steps512 | repeat 1, 512 steps/world/iter, minibatch 1024, 450 its | control: is repeat 4's gain only from seeing 4x more frames per iteration? |
| r2_rep4_uniform01 | repeat 4, difficulty uniform 0..1 per episode, no curriculum | is the curriculum needed at all? |

Added `--sample-difficulty upto` (each episode U(0, curriculum level)) to train.py.
- r1_blockonly_repeat4 final: 0.43/0.39/0.17/0.06/0.04/0.02 | 0.98. It masters blocker levels (98%) but still
  transfers only partly to normal levels (43% at difficulty 0). Same lesson as r1_blockonly, but faster.
- Early round 2 (~110 iterations): r2_rep4_rays16 is far ahead (fixed 0.2: 79%, blocked 57%, clear 91%);
  r2_rep4_mix50_d0 is also good on blocked levels (57%).

## Round 3

| run | change | question |
|---|---|---|
| r3_rep4_rays16_mix50_d0 | repeat 4 + rays 16 + blocker 0.5 + start 0 | do the two winners add up? |
- r1_repeat4 final: 0.81/0.67/0.30/0.14/0.04/0.00 | 0.42. **Best of round 1 by far** (baseline 0.67/0.41/0.23/...
  | 0.01). Repeat 4 alone takes blocker-level success from 1% to 42%. Its curriculum also never left 0.2 though.
| r3_rays16_rep1 | rays 16, repeat 1 | ablation: do rays help without repeat? |
- r2_rep4_uniform01 final: 0.41/0.21/0.04/0.00/0.00/0.00 | 0.02. Uniform 0..1 difficulty with no curriculum is
  much worse: too many hard levels early, so nothing gets learned well. **Curriculum needed.**
- ~06:18, iteration ~375: r3_rep4_rays16_mix50_d0 is the clear leader: fixed 0.2 85% (blocked 78%), fixed 0.5
  72% (blocked 70%), curriculum at 0.45 and climbing. Baseline's fixed 0.5 was 12%.
| r3_combo_seed1 | same as r3_rep4_rays16_mix50_d0, seed 1 | is the combo's result reproducible? |
- r2_rep4_mix50_d0 final: 0.81/0.76/0.59/0.28/0.14/0.01 | 0.93. Blocker mix + repeat 4 (no rays): big gain at
  0.2-0.6 over r1_repeat4 (0.81/0.67/0.30/0.14/...), and it now dodges (93% on blocker levels).

## Round 4

| run | change | question |
|---|---|---|
| r4_combo_long | combo (repeat 4, rays 16, blocker 0.5, start 0), 3000 iterations, fixed evals at .2/.5/.8 | how far can it go with more time? |
- r2_rep4_rays16 final: 0.83/0.85/0.65/0.28/0.12/0.01 | 0.45.
- r3_rep4_rays16_mix50_d0 (the "combo") final: 0.67/0.90/0.78/0.56/0.41/0.23 | 0.84. Best so far at 0.2+.
- r2_control_steps512 final: 0.88/0.60/0.33/0.15/0.04/0.01 | 0.03. **Control: 4x frames per iteration with repeat 1
  does NOT reproduce repeat 4's dodging (3% vs 42% on blocker levels).** The gain is the decision horizon.
- r3_rays16_rep1 final: 0.79/0.64/0.36/0.15/0.06/0.01 | 0.13. Rays help a bit alone, a lot together with repeat.

### Bug found: observation normalization breaks on features that never varied in training

`experiments/analyze.py` on the combo checkpoint (iteration 500, curriculum at 0.5) at difficulty 0.8:
levels with 6-8 planets: **1% success (97% crash)** vs 55% with 4-5 planets. Cause: at difficulty <= 0.5 a level
has at most round(2 + 6*0.5) = 5 planets, so obstacle slots 6-8 were always empty in training -> RunningMeanStd
std = 0 exactly -> a real planet in slot 6 becomes (x - 0) / 1e-4, clipped to +-10. Garbage input.
Proof: copying slot 5's statistics into slots 6-8 of the checkpoint (no retraining) raised 6+ planet levels
from 1% to 28% and overall success at 0.8 from 38% to 50%.

The same bug in reverse explains the forgetting at difficulty 0 (combo final: 57% at 0.0 but 89% at 0.2):
151 of 170 crashes at difficulty 0 are into walls, on levels with 0-1 planets (43% success vs 94% with 2-3).
In training slot 1 was almost always filled (presence mean 0.987, std 0.11), so an EMPTY slot 1 normalizes to
(0 - 0.987)/0.11 = -9.

This bug is in the user's outerworldrl.py too. In a curriculum it bites every time a promotion unlocks a new
planet count. Fix candidates: no normalization (the C++ observation is already scaled to ~+-1), or a std floor.

| run | change | question |
|---|---|---|
| r4_combo_h128 | combo + 128 hidden | network size |
| r4_combo_nonorm | combo + no obs normalization | does fixing the normalization bug fix 6+ planets and difficulty 0? |
| r4_combo_nonorm_seed1 | same, seed 1 | replicate |
| r4_combo_nonorm_upto | + per-episode difficulty U(0, level) | does mixing in easier levels help further? |
- r3_combo_seed1 final: 0.76/0.89/0.79/0.55/0.39/0.13 | 0.95. **The combo reproduces across seeds.**

## Round 5
| run | change | question |
|---|---|---|
| r5_nonorm_rep8 | combo + no norm, repeat 8 | is a longer decision horizon even better, or does it cost precision? |
- r4_combo_h128 final: 0.65/0.88/0.79/0.56/0.38/0.20 | 0.89. Same as 64 hidden (0.67/0.90/0.78/0.56/0.41/0.23)
  within noise: **network size is not the bottleneck** (in this setup; it still has the norm bug).
| r5_nonorm_straight | combo + no norm, straight-line shaping instead of path | is the distance field worth it? |
- **r4_combo_nonorm final: 0.93/0.90/0.85/0.72/0.52/0.34 | 0.95. Best everywhere.** Turning off obs normalization
  fixed difficulty 0 (67% -> 95%, the wall crashes are gone) and 6+ planet levels at 0.8 (1% -> 35%).
- Stopped r4_combo_long at iteration ~800 (it has the norm bug); replaced by r6_best_long.

## Round 6
| run | change | question |
|---|---|---|
| r6_best_long | best config (repeat 4, rays 16, blocker 0.5, start 0, no norm), 4000 iterations | how far with time? |
| r5_nonorm_klepoch | best config + KL early stop per epoch on the whole batch | does it help now? |
- r4_combo_nonorm_seed1 final: 0.95/0.88/0.78/0.59/0.42/0.23 | 0.96. Replicates (seed spread ~0.1 at 0.8-1.0, so
  treat smaller differences at high difficulty as noise).
- r4_combo_nonorm_upto final: 0.94/0.93/0.85/0.62/0.46/0.28 | 1.00. Between the two plain seeds: no clear gain.
  Its curriculum reached 0.90 (vs 0.55-0.60) only because easier sampled levels raise train success.
- Analysis of r4_combo_nonorm at difficulty 1.0: 37% overall; 4-5 planets 60%, 6-8 planets 19%. Gravity on every
  level, turn angle / distance / goal size matter little. The agent reached curriculum ~0.6 (max 6 planets), so
  7-8 planets are unseen: with a per-slot MLP an unseen slot is effectively untrained.

## Round 7: Deep Sets

`experiments/models.py`: `--arch deepsets` runs each obstacle slot through one shared network phi
(5 -> 32 -> 32, ReLU), max-pools over present planets, concatenates with the other 14 + 16 ray values, then
the usual 64x64 MLP. 9.8k parameters vs 9.1k for the MLP.

| run | change | question |
|---|---|---|
| r7_deepsets, r7_deepsets_seed1 | best config + deepsets | does a permutation-invariant planet encoder generalize to more planets? |
- r5_nonorm_straight final: 0.97/0.90/0.84/0.71/0.51/0.30 | 0.98, equal to path shaping
  (0.93/0.90/0.85/0.72/0.52/0.34) within noise. **With blocker training + rays + repeat, the distance field
  shaping no longer matters.** (Answers the user's "straight vs path" experiment.)
| r5_nonorm_rep8_seed1 | repeat 8, seed 1 | replicate: repeat 8 looks strong at iteration 518 (fixed .5 86%) |
- **r5_nonorm_rep8 final: 0.94/0.92/0.89/0.76/0.66/0.56 | 0.97.** Clearly better than repeat 4
  (0.93/0.90/0.85/0.72/0.52/0.34): +0.14 at 0.8, +0.22 at 1.0, beyond seed noise. At 1.0: 4-5 planets 74%,
  6-8 planets 43% (repeat 4: 60% / 19%).

## Round 8: how far does action repeat go?
| run | change | question |
|---|---|---|
| r8_rep16 | best config, repeat 16 (0.27 s per decision) | still better, or too coarse to steer? |
- r5_nonorm_klepoch final: 0.90/0.91/0.88/0.74/0.49/0.28 | 0.96. Same as minibatch-KL. At minibatch 256 the early
  stop almost never fires anyway (mean epochs completed 9.99 vs 9.88), so **the KL stop mode doesn't matter here**.
| r8_rep8_gate70 | repeat 8, promote at 70% instead of 80% | is the curriculum gate what limits progress to hard levels? |
- **Determinism check:** r6_best_long (same seed and settings as r4_combo_nonorm) matched it exactly at every fixed
  eval through iteration 250. Training is fully reproducible for a given seed.
- Both repeat-8 seeds beat both repeat-4 seeds at fixed 0.5 at equal iterations (100-250), and repeat 8's final
  result is much better at 0.8-1.0, so I stopped r6_best_long (repeat 4, ~650 its; its first 600 duplicate
  r4_combo_nonorm anyway) and started **r9_rep8_long**: repeat 8, 4000 iterations.
- r5_nonorm_rep8_seed1 final: 0.94/0.94/0.89/0.74/0.52/0.44 | 0.99. **Repeat 8 replicates**: at 0.8/1.0 the two
  repeat-8 seeds got 0.66/0.56 and 0.52/0.44 vs repeat-4 seeds 0.52/0.34 and 0.42/0.23.
- **r7_deepsets final: 0.93/0.94/0.85/0.77/0.71/0.61 | 0.98; seed 1: 0.96/0.92/0.84/0.73/0.65/0.53 | 0.94.**
  Deep Sets (with repeat 4) beats the MLP with repeat 4 by ~+0.2 at 0.8 and ~+0.28 at 1.0, on both seeds.
  At 1.0: 6-8 planets 49% (MLP: 19%), 4-5 planets 76% (60%). Shared per-planet weights generalize to more planets.

## Round 10
| run | change | question |
|---|---|---|
| r10_ds_rep8, r10_ds_rep8_seed1 | deepsets + repeat 8 | do the two best changes add up? |
- r8_rep16 final: 0.91/0.85/0.81/0.57/0.41/0.26 | 0.65. Worse than 8 (0.94/0.92/0.89/0.76/0.66/0.56): 16 frames
  (0.27 s) per decision is too coarse. **Repeat 8 is the sweet spot: 4 < 8 > 16.**
- Started r11_ds_rep8_long (deepsets + repeat 8, 4000 iterations) before r10 finished, betting the two add up.
| r10_ds_rep8_norays | deepsets + repeat 8, no rays | ablation: with deepsets, are the rays still needed? |
- r8_rep8_gate70 final: 0.93/0.88/0.84/0.70/0.64/0.47 | 0.90. Between the two 80%-gate repeat-8 seeds: the
  curriculum climbed further (0.75 vs 0.60) but the agent isn't better on hard levels. **The gate is not the
  limit**, and the curriculum level by itself overstates progress; trust the fixed evals.
| r12_ds_rep8_ent003 | deepsets + repeat 8, entropy 0.003 | is the entropy bonus holding back precise control? |
- **r10_ds_rep8 final: 0.90/0.90/0.86/0.85/0.76/0.73 | 0.96; seed 1: 0.96/0.94/0.90/0.82/0.76/0.68 | 0.98.
  Deep Sets + repeat 8 is the best config, on both seeds.** (Original baseline: 0.67/0.41/0.23/0.08/0.04/0.03.)
- **Generalization check** (r10_ds_rep8 final, 1000 levels each, greedy): held-out vs training levels:
  0.2: 89% vs 90%, 0.5: 84% vs 86%, 0.8: 80% vs 79%, 1.0: 72% vs 69%. No memorization at all.
- Remaining failures at 1.0 (1000 levels): fairly uniform; 4-5 planets 81%, 6-8 planets 67%; turn angle,
  distance, goal size matter little. 202 planet crashes, 44 wall crashes, 29 timeouts.
| r12_ds_rep8_blocker25 | blocker 0.25 instead of 0.5 | how much blocker mixing is needed? |
| r12_ds_rep8_envs64 | 64 worlds (8192 decisions per iteration) | more level diversity per update (compare at equal frames) |
- Added `experiments/replay.py` (records a checkpoint on held-out levels as viewer replay files). Saved 12 replays
  of r10_ds_rep8 at difficulty 1.0 in runs/exp/r10_ds_rep8/replays/ (8 goals, 3 crashes, 1 timeout). Verified
  each re-simulates one frame at a time (as the viewer does) to the same result and length.
- r10_ds_rep8_norays final: 0.96/0.92/0.82/0.88/0.67/0.58 | 0.99. Similar up to 0.6, ~0.1 lower at 0.8-1.0 than
  both with-rays seeds (0.76/0.73, 0.76/0.68). **Rays still help on the hardest levels** (one seed, modest).
| r12_ds_rep8_lr1e3 | lr 1e-3 instead of 3e-4 | faster learning without instability? |

## Ported the best config into python/outerworldrl.py (~07:37)

Backup of the user's version: experiments/backup/outerworldrl_before_overnight.py.
Changes: ACTION_REPEAT 8, RAYS 16, BLOCKER_CHANCE 0.5, start difficulty 0, eval env with same repeat/rays (no
blockers), OBS = env.obs_size, RunningMeanStd removed (comment explains why), PlanetSetNet (Deep Sets) for policy
and value, line_blocked slices 14:54, checkpoint stores "args" so analyze.py / replay.py can rebuild the network
(models.load_state maps PlanetSetNet's "mlp." names to DeepSets' "rho."). Smoke-tested 3 iterations.
- r12_ds_rep8_ent003 final: 0.99/0.94/0.89/0.76/0.67/0.54 | 0.94. Slightly worse on hard levels than ent 0.01
  (0.76/0.73, 0.76/0.68). **Keep 0.01.**
| r13_user_script | python/outerworldrl.py as ported (500 iterations), logs to runs/exp/r13_user_script | validates the port |
- r12_ds_rep8_lr1e3 final: 0.98/0.92/0.87/0.85/0.72/0.61 | 0.99. Same as 3e-4 within noise.
- Long runs, fixed evals at 0.2/0.5/0.8 (100 levels each):
  - r9_rep8_long (MLP): plateau. 800: 0.90/0.86/0.70, 1400: 0.88/0.87/0.70, 1800: 0.96/0.88/0.75; curriculum stuck
    at 0.75 from iteration ~1000. Stopped at ~1870 to free the slot.
  - r11_ds_rep8_long (Deep Sets): still climbing. 600: 0.87/0.85/0.73, 800: 0.87/0.88/0.78, 1000: 0.94/0.94/0.92,
    curriculum 0.90.
| r14_ds_rep8_long_seed1 | Deep Sets long run, seed 1, 3000 its | replicate the long run |
| r14_ds_rep8_big | Deep Sets, hidden 128, embed 64 | capacity, now that the other bottlenecks are gone |
- r12_ds_rep8_blocker25 final: 0.95/0.96/0.92/0.84/0.81/0.74 | 0.98. As good as blocker 0.5 (0.76/0.73,
  0.76/0.68 at 0.8/1.0), maybe slightly better; replicating.
| r12_ds_rep8_blocker25_seed1 | blocker 0.25, seed 1 | replicate |
- **r13_user_script (python/outerworldrl.py as ported), checkpoint at iteration 450:
  0.96/0.94/0.94/0.88/0.80/0.72 | 0.98** (200 held-out levels each, same levels as every results.json).
  Matches r10_ds_rep8 / seed 1. **The port works.**
- r14_ds_rep8_big (hidden 128, embed 64) final: 0.96/0.92/0.87/0.81/0.72/0.67 | 0.99. Same as small. Capacity
  still not the bottleneck.
- r11_ds_rep8_long: curriculum reached 1.00 at ~1400 iterations; fixed 0.2/0.5/0.8 at 1800: 0.93/0.96/0.93.
- Added `--pool maxmean` (models.py): max and mean over present planets, concatenated.
| r15_ds_rep8_maxmean | deepsets with max+mean pooling | does a "how crowded" summary help? |
- Long runs at iteration 1850, final-eval style (200 held-out levels, 0/.2/.4/.6/.8/1):
  - r9_rep8_long (MLP, stopped): 0.97/0.95/0.93/0.86/0.80/0.73
  - r11_ds_rep8_long (Deep Sets): 0.90/0.92/0.90/0.88/0.92/0.88. **88% at difficulty 1.0.** Slight dip at 0
    (curriculum has been at 1.0 since ~1400, maybe forgetting the easiest levels a little).
- **r12_ds_rep8_envs64 final: 0.97/0.96/0.94/0.92/0.91/0.90 | 0.96** (600 its = 2x the frames of a 32-world run,
  40 min). At equal wall-clock (~40 min = r11 at ~1200 its) r11's fixed 0.2/0.5 were 0.89/0.90 vs 0.96/0.93 here.
  More levels per update looks better at equal compute (one seed so far).
| r12_ds_rep8_envs64_seed1 | 64 worlds, seed 1 | replicate |
- r12_ds_rep8_blocker25_seed1 final: 0.99/0.98/0.93/0.80/0.75/0.56 | 0.95. Two-seed mean at 0.8/1.0: blocker 0.25
  0.78/0.65 vs blocker 0.5 0.76/0.71. Same within noise; keep 0.5.
| r16_ds_rep8_envs64_long | 64 worlds, 2000 its, fixed evals .2/.5/.8/1.0 | the strongest agent we can train tonight |
- r10_ds_rep8_norays_seed1 final: 0.99/0.98/0.95/0.90/0.74/0.72 | 0.99. **Corrects the earlier conclusion**: two-seed
  means at 0.8/1.0: no rays 0.71/0.65, rays 0.76/0.71. ~0.05, within noise. With Deep Sets the rays are optional
  (kept in the user's script; cheap).
| r15_ds_rep8_shaping_none | no progress shaping (goal/crash rewards only) | completes the user's straight/path/none question |
- r15_ds_rep8_maxmean final: 0.98/0.94/0.92/0.89/0.82/0.71 | 0.98. Same as max pooling within noise; keep max.
- r14_ds_rep8_long_seed1 reached curriculum 1.0 at ~1290 iterations (r11: ~1400). Replicates.
| r17_ds_rep8_upto_long | long run with difficulty ~ U(0, level) per episode, fixed evals at 0/.5/1 | does it stop the forgetting at difficulty 0? |
- r15_ds_rep8_shaping_none final: 0.97/0.94/0.93/0.83/0.70/0.56 | 0.91. Only goal/crash rewards, and it still learns
  almost as well (path shaping two-seed mean at 0.8/1.0: 0.76/0.71). **Shaping helps a little on the hardest levels
  but isn't essential** once repeat 8 + curriculum are in. Completes the user's straight/path/none question.
| r15_ds_rep8_epochs4 | 4 PPO epochs instead of 10 | the update is the bottleneck: same quality for less time? |
- r15_ds_rep8_epochs4 final: 0.98/0.93/0.81/0.68/0.56/0.44 | 0.93 in 13.5 min (10 epochs: 0.76/0.70 at 0.8/1.0 in
  23 min). Worse per iteration, not better per minute. Keep 10 epochs.
| r15_ds_rep8_lam90 | lambda 0.90 | with repeat 8 the horizon is long already; does lower lambda (less variance) help? |
- r12_ds_rep8_envs64_seed1 final: 0.98/0.97/0.94/0.91/0.89/0.83 | 0.99. **64 worlds replicates.** Also better at
  equal wall-clock (32-world long run r11 at ~40 min / 1200 its: fixed 0.8 = 0.83; 64 worlds at 40 min: 0.89-0.91).
  Set NUM_ENVS = 64 in python/outerworldrl.py.
- experiments/plot_progress.py -> experiments/progress.png (held-out success vs difficulty per improvement step).
- **r11_ds_rep8_long final (4000 its): 0.94/0.94/0.93/0.93/0.94/0.93 | 0.97.** ~93% at every difficulty. Fixed evals
  hover 0.92-0.96 from ~1000 iterations on. On 1000 levels at 1.0: 90% greedy, 88% sampled.
  Replays: runs/exp/r11_ds_rep8_long/replays (12 successes), replays_failures (15 failures of the first 200 levels).
- r15_ds_rep8_lam90 final: 0.95/0.94/0.91/0.76/0.76/0.67 | 0.88. Same as 0.95 within noise.
- **Greedy deadlock found:** of r11's failures at 1.0, 35/95 (1000 levels) are timeouts, and the ship *never moves*:
  either action 0 forever (no thrust, no turn) or turning left/right alternately (a period-2 loop). Spawn is
  outside gravity, so if argmax says "do nothing" the observation never changes and it never will. Sampling the
  same policy solves 55/70 attempts on those 7 levels. Cause: waiting is free - a timeout bootstraps V(spawn),
  and nothing penalizes time. Added `--step-penalty` (subtracted every decision).
| r18_ds_rep8_steppen01, r18_ds_rep8_steppen03 | step penalty 0.01 / 0.03 per decision | does a time cost remove the deadlocks without hurting? |
- **r14_ds_rep8_long_seed1 final (3000 its): 0.99/0.95/0.93/0.97/0.90/0.91 | 0.98.** The long-run result replicates.
| r18_ds_rep8_timeout_terminal | timeouts not bootstrapped (worth 0) | the other fix for "waiting is free" |
- r12_ds_rep8_envs64_seed2 final: 0.96/0.94/0.94/0.92/0.91/0.88 | 0.99. Three seeds at 1.0: 0.90/0.83/0.88 (mean 0.87).
| r19_ds_rep8_envs128 | 128 worlds, 300 iterations (same frames as 64 x 600) | does it keep scaling? |
- r18_ds_rep8_steppen01 final: 0.98/0.96/0.94/0.84/0.74/0.62 | 0.98; timeouts at 1.0: 10% (vs 2-4% without).
  No help at 600 iterations (deadlocks are not the main failure there; they show up in the long runs).
| r20_user_script_64 | python/outerworldrl.py as it now stands (64 worlds), 500 its | end-to-end check + a trained checkpoint for the user |
- r18_ds_rep8_steppen03 final: 0.96/0.92/0.91/0.89/0.81/0.72 | 0.97; timeouts 1-3%. As good as no penalty or
  slightly better; replicating.
| r18_ds_rep8_steppen03_seed1 | step penalty 0.03, seed 1 | replicate |
- r18_ds_rep8_timeout_terminal final: 0.94/0.92/0.85/0.78/0.78/0.62 | 0.97. No better, more crashes. Not recommended.
| r21_envs64_steppen03_long | 64 worlds, step penalty 0.03, 1500 its | the deadlocks show up in long runs: does the penalty remove them there? (compare r16) |
- r17_ds_rep8_upto_long final (2000 its): 0.97/0.99/0.94/0.91/0.91/0.83 | 1.00. Better on easy levels, worse at 1.0
  than the plain long runs (0.93, 0.91): it trades hard-level skill for easy-level retention. Not recommended.
| r22_ds_rep12 | Deep Sets, repeat 12 | refine the repeat sweet spot (16 was only tested with the MLP) |
- **r16_ds_rep8_envs64_long final (2000 its, 103 min): 0.98/0.96/0.96/0.95/0.94/0.93 | 0.99. Best agent.**
  1000 levels at 1.0: 92% (64 crashes: 25 walls, 39 planets; 14 timeouts vs 35 for r11).
| r16_seed1 | replicate of the best long run, 1500 its | |
- r18_ds_rep8_steppen03_seed1 final: 0.98/0.94/0.89/0.82/0.76/0.67 | 0.99. Two-seed mean at 0.8/1.0: 0.79/0.70 vs
  0.76/0.71 without penalty: no difference at 600 its. Verdict waits for r21 (long run, where deadlocks appear).
- r19_ds_rep8_envs128 final (300 its = same frames as 64 x 600, 38 min): 0.97/0.96/0.89/0.90/0.83/0.73 | 0.95.
  Worse than 64 worlds (0.83-0.90 at 1.0): half as many updates. 64 is the sweet spot at this budget.
- r20_user_script_64 (python/outerworldrl.py as it stands, 64 worlds), checkpoint at 450: 0.96/0.95/0.94/0.89/0.84/0.78 | 0.98.
| r23_ds_rep8_noblocker | final setup (32 worlds), blocker 0 | are blocker levels still needed with Deep Sets + repeat 8? |
| r23_ds_rep8_nocurriculum | final setup, difficulty uniform 0..1, no curriculum | is the curriculum still needed? |
- r22_ds_rep12 final: 0.98/0.94/0.94/0.87/0.71/0.65 | 0.98. Same as repeat 8 or slightly worse. Keep 8.
- r23_ds_rep8_nocurriculum final: 0.95/0.95/0.92/0.83/0.68/0.68 | 0.99. Same as with curriculum (0.76/0.71 at
  0.8/1.0). With the weak early setup no-curriculum was a disaster (r2_rep4_uniform01: 0.41 at difficulty 0);
  with the final setup **the curriculum is no longer needed** (one seed; replicating).
| r23_ds_rep8_nocurriculum_seed1 | replicate | |
- r23_ds_rep8_noblocker final: 0.91/0.91/0.84/0.78/0.60/0.55 | 0.88 vs 0.76/0.71 at 0.8/1.0 with blockers and
  96-98% on blocker levels. **Blocker levels still help (~0.15 on hard levels) even with the final setup.**
- r23_ds_rep8_nocurriculum_seed1 final: 0.94/0.90/0.84/0.76/0.69/0.63 | 0.97. Two-seed mean 0.69/0.66 vs
  curriculum 0.76/0.71: the curriculum helps a little (~0.05) but is no longer essential.
| r24_best_continued | r16's final.pt trained 1000 more its at difficulty 1.0 (fresh Adam), fixed evals .5/1.0 on 200 levels | does more training keep helping past 93%? |
- r23_ds_rep8_noblocker_seed1 final: 0.91/0.86/0.83/0.74/0.60/0.51 | 0.72. Confirms: blockers still matter.

## Zero-shot test: more planets than ever seen (~11:15)

Built a scratch copy of the project (scratchpad/ow12, the repo's source untouched) with OBS_MAX_OBSTACLES = 12,
so the observation has 12 slots and the generator puts up to 12 planets at difficulty 1.0. models.py / analyze.py
now take the slot count from the DLL (defaults unchanged; regression check: r10 still 73% at 1.0).
Best agent (r16, trained on <= 8 planets), 1000 levels at difficulty 1.0, no retraining:
- sees all 12 slots: 85% overall; 4-5 planets 96%, 6-8 91%, **9-10: 83%, 11-12: 69%**.
- sees only the nearest 8 (slots 9-12 zeroed): 84%; buckets within 1-3 points.
So it generalizes to denser levels than any in training (the MLP can't even take a 12-slot input), and planets
beyond the nearest 8 rarely matter; the drop on 11-12 planets is the levels being harder (crashes), not perception.
- r21_envs64_steppen03_long final (1500 its): 0.97/0.95/0.91/0.91/0.92/0.90 | 0.99. 1000 levels at 1.0: 88%
  (6 timeouts, 112 crashes) vs r16 (no penalty, 2000 its) 92% (14 timeouts, 64 crashes). Fewer deadlocks, more
  crashes (it hurries). No net gain: not recommended.
- r16_ds_rep8_envs64_long_seed1 final (1500 its): 0.98/0.95/0.95/0.93/0.94/0.88 | 0.98. Replicates.
- r24_best_continued (r16 + more its at 1.0): fixed 1.0 on 200 levels hovers 0.94-0.97 over 450 its; plateau.
- `analyze.py --unstick N`: greedy, but sample in worlds whose ship hasn't moved for N decisions. Best agent (r16),
  1000 levels: 0.5: 96% -> 98%, 1.0: 92% -> 94%, timeouts 21/14 -> 0/0, crashes unchanged. The deadlock is purely
  a greedy-evaluation artifact.
| r25_user_script_long | python/outerworldrl.py as-is, 2000 iterations | a strong agent trained by the user's own script |
- r24_best_continued final: 0.99/0.98/0.95/0.95/0.94/0.91 | 0.99. 1000 levels at 1.0: greedy 90% (54 timeouts,
  43 crashes), greedy + unstick 8: **95%** (47 crashes). More training cuts crashes but adds deadlocks.
- r25_user_script_long (the user's python/outerworldrl.py, 2000 its, latest.pt): 0.96/1.00/0.98/0.96/0.96/0.96 | 1.00
  (200 levels). 1000 levels: 0.5 97% (3 timeouts), 1.0 94% greedy (0 timeouts, 62 crashes), unstick 94%.
  Same 1000 levels: r16 92% greedy / 94% unstick, r24 90% / 95%. Best greedy agent; the repo script reproduces the best recipe.
- Added `train.py --anneal-lr` (lr decays linearly from --lr to 0 over the run; the standard PPO trick, untried so far).
  Running r26_envs64_anneal_long (seed 0) and _seed1: r16's exact config + --anneal-lr, 2000 its, both at once.
  Question: does settling the policy at the end cut the ~6% crashes at difficulty 1.0?
- r26 anneal (2000 its) finals: seed0 0.98/0.96/0.93/0.96/0.96/0.96 | 0.99, seed1 0.99/0.98/0.98/0.97/0.96/0.95 | 0.98.
  1000 levels at 1.0, greedy / unstick: seed0 94.3/94.5 (2 timeouts), seed1 94.8/95.4 (8). vs r16 92.2/93.6. Small consistent gain.
- Failure overlap (analyze.py --dump, runs/exp/overlap/): 4 agents fail 46-64 each, pairwise shared 18-28, all-4 only 10.
  Oracle-any 99.0%. r26s1 sampled x4: solves 36/46 of its greedy failures. Never solved by anything: 6 levels
  (265, 523, 111, 657, 213, 214). => failures are agent-specific, not level-intrinsic.
- analyze.py --ensemble (mean of softmax probs): 4-agent 0.5 98.7%, 1.0 96.5% (unstick same); 2 annealed agents
  1.0 96.9%. Ensembles +~2 points; 2 members enough.
- Running r27_envs64_anneal_4000 (anneal, 4000 its, seed 0).
- experiments/distill.py (DAgger-style: student samples its own actions on TRAIN levels, teachers label visited
  states with mean probs, buffer 200k, cross-entropy to soft targets, Adam 1e-4, 100 x 1024 updates/round, ~2.3 s/round).
  Teachers = r26 seed0 + seed1. 1000 held-out levels (0.0 / 0.5 / 1.0 | blocker):
  - d1_ens2_init (init r26s1, 300 rounds ~12 min): 99.2 / 98.7 / **96.2** | 99.5. Matches the 2-ensemble in one net.
  - d2_ens2_fresh (random init, 600 rounds ~25 min): 99.4 / 96.8 / 95.4 | 99.3. = best PPO agent, from scratch.
  - Running d3_self_control: teacher = r26s1 alone, init r26s1 (is the gain the ensemble or just more low-lr training?).
  - d3_self_control: 0.5 97.3%, 1.0 94.7% (= the teacher itself, 97.3/94.8). So d1's +1.4 is the ensemble's knowledge.
- Round 2: ens(d1, d2, r26s0) 96.7% at 1.0; distilled d4_round2 98.3 / 96.4. Plateau ~96-97%.
- Never-solved levels, d1 sampled 64x: 265 0/64, 523 22, 111 29, 657 3, 213 5, 214 9. Only 265 looks impossible.
