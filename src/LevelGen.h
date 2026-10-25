#pragma once
#include <cstdint>
#include <numbers>
#include "constants.h"
#include "Level.h"

// Settings for the random level generator. Every range is inclusive.
struct GenParams {
    int minObstacles = 3;
    int maxObstacles = 6;
    float minRadius = 0.08f;
    float maxRadius = 0.25f;
    float gravityChance = 0.5f;                                 // chance that an obstacle has gravity
    float minDensity = 0.5f * constants::PLANET_DENSITY;        // used when it does
    float maxDensity = 1.5f * constants::PLANET_DENSITY;
    float minSpawnGoalDistance = 1.5f;
    float corridorFraction = 0.5f;                              // share of obstacles placed near the spawn-goal line
    float corridorWidth = 0.4f;                                 // how far from that line they can be
    float minGap = 0.2f;                                        // free space between obstacles (ship is 0.08 wide)
    float spawnAngleJitter = std::numbers::pi_v<float>;         // 0 = facing the goal, pi = any direction
    float minGoalHalfSize = 0.06f;
    float maxGoalHalfSize = 0.2f;
};

// 0 = easy (few obstacles, no gravity, goal close and facing it), 1 = hard. For curriculum learning.
GenParams paramsForDifficulty(float difficulty);

// Builds a random level. The same seed and params always give the same level, on any compiler.
//
// Every level is solvable as far as obstacles go: obstacles keep a ship-sized gap from each other
// and from the edges, so the free space is always connected. Gravity is limited near the spawn
// and the goal, but strong wells can still make some routes much harder.
Level generateLevel(uint32_t seed, const GenParams& params);
