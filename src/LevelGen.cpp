#include "LevelGen.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace {

constexpr float SPAWN_MARGIN = 0.25f;     // how far the spawn and goal stay from the edges
constexpr float SPAWN_CLEARANCE = 0.3f;   // free space around the spawn
constexpr float GOAL_CLEARANCE = 0.2f;    // free space around the goal
constexpr float EDGE_MARGIN = 0.15f;      // obstacles keep this far from the edges, so ships can pass along them
constexpr float MAX_GOAL_PULL = 0.75f;    // gravity at the goal is kept below this fraction of thrust
constexpr int ATTEMPTS_PER_OBSTACLE = 200;
constexpr float BLOCKER_MIN_RADIUS = 0.12f;  // a blocker is at least 3x the ship's width

// std::uniform_real_distribution gives different numbers on different compilers, so the raw
// mt19937 output (which is the same everywhere) is turned into floats here instead.
class Rng {
public:
    explicit Rng(uint32_t seed) : engine(mix(seed)) {}

    float uniform(float lo, float hi) {
        float unit = (float)(engine() >> 8) * (1.0f / 16777216.0f);   // 24 random bits -> [0, 1)
        return lo + (hi - lo) * unit;
    }

    int uniformInt(int lo, int hi) {   // inclusive
        int value = lo + (int)uniform(0.0f, (float)(hi - lo + 1));
        return std::min(value, hi);
    }

    bool chance(float p) { return uniform(0.0f, 1.0f) < p; }

private:
    // Scrambles the seed so that nearby seeds (1, 2, 3...) give unrelated levels.
    static uint32_t mix(uint32_t x) {
        x ^= x >> 16; x *= 0x7feb352d;
        x ^= x >> 15; x *= 0x846ca68b;
        x ^= x >> 16;
        return x;
    }

    std::mt19937 engine;
};

float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

// Distance from a point to the nearest point of the goal rectangle.
float distanceToGoal(glm::vec2 p, const Goal& goal) {
    glm::vec2 closest = glm::clamp(p, goal.position - goal.halfSize, goal.position + goal.halfSize);
    return glm::distance(p, closest);
}

bool obstacleFits(const Obstacle& o, const Level& level, const GenParams& params) {
    // Inside the world, with room to fly between it and the edge
    if (std::abs(o.position.x) + o.radius > constants::WORLD_HALF_WIDTH - EDGE_MARGIN) return false;
    if (std::abs(o.position.y) + o.radius > constants::WORLD_HALF_HEIGHT - EDGE_MARGIN) return false;

    // Clear of the spawn, and the spawn outside its gravity
    float toSpawn = glm::distance(o.position, level.spawn);
    if (toSpawn < o.radius + SPAWN_CLEARANCE) return false;
    if (o.density > 0.0f && toSpawn < o.gravityRange() + constants::SHIP_RADIUS) return false;

    // Clear of the goal, and not pulling too hard on it
    float toGoal = distanceToGoal(o.position, level.goal);
    if (toGoal < o.radius + GOAL_CLEARANCE) return false;
    if (o.density > 0.0f) {
        glm::vec2 nearestGoalPoint = glm::clamp(o.position, level.goal.position - level.goal.halfSize,
                                                level.goal.position + level.goal.halfSize);
        glm::vec2 pull = Obstacle::gravityAt(nearestGoalPoint, {o});
        if (glm::length(pull) > MAX_GOAL_PULL * constants::SHIP_THRUST) return false;
    }

    // A ship-sized gap from every other obstacle
    float gap = std::max(params.minGap, 3.0f * constants::SHIP_RADIUS);
    for (const auto& other : level.obstacles) {
        if (glm::distance(o.position, other.position) < o.radius + other.radius + gap) return false;
    }
    return true;
}

}  // namespace

GenParams paramsForDifficulty(float difficulty) {
    float d = std::clamp(difficulty, 0.0f, 1.0f);
    GenParams p;
    p.minObstacles = (int)std::round(lerp(0.0f, 4.0f, d));
    p.maxObstacles = (int)std::round(lerp(2.0f, (float)constants::OBS_MAX_OBSTACLES, d));
    p.minRadius = 0.08f;
    p.maxRadius = lerp(0.18f, 0.28f, d);
    p.gravityChance = lerp(0.0f, 0.7f, d);
    p.minDensity = lerp(0.5f, 1.0f, d) * constants::PLANET_DENSITY;
    p.maxDensity = lerp(0.8f, 1.5f, d) * constants::PLANET_DENSITY;
    p.minSpawnGoalDistance = lerp(0.8f, 2.6f, d);
    p.corridorFraction = lerp(0.2f, 0.7f, d);
    p.corridorWidth = 0.4f;
    p.minGap = lerp(0.35f, 0.15f, d);
    p.spawnAngleJitter = lerp(0.5f, std::numbers::pi_v<float>, d);
    p.minGoalHalfSize = lerp(0.12f, 0.05f, d);
    p.maxGoalHalfSize = lerp(0.25f, 0.15f, d);
    return p;
}

Level generateLevel(uint32_t seed, const GenParams& params) {
    Rng rng(seed);
    Level level;
    const float W = constants::WORLD_HALF_WIDTH;
    const float H = constants::WORLD_HALF_HEIGHT;

    // Goal (a rectangle fully inside the world) and spawn, chosen as a pair so they can be far apart.
    // Keeps the farthest pair in case none reaches the minimum distance.
    float bestDistance = -1.0f;
    for (int attempt = 0; attempt < 200; ++attempt) {
        Goal goal;
        goal.halfSize = {rng.uniform(params.minGoalHalfSize, params.maxGoalHalfSize),
                         rng.uniform(params.minGoalHalfSize, params.maxGoalHalfSize)};
        glm::vec2 goalRange = glm::vec2(W, H) - SPAWN_MARGIN - goal.halfSize;
        goal.position = {rng.uniform(-goalRange.x, goalRange.x), rng.uniform(-goalRange.y, goalRange.y)};
        glm::vec2 spawn{rng.uniform(-W + SPAWN_MARGIN, W - SPAWN_MARGIN), rng.uniform(-H + SPAWN_MARGIN, H - SPAWN_MARGIN)};

        float distance = distanceToGoal(spawn, goal);
        if (distance > bestDistance) {
            level.goal = goal;
            level.spawn = spawn;
            bestDistance = distance;
        }
        if (distance >= params.minSpawnGoalDistance) break;
    }

    // Facing the goal, plus a random turn. Forward is (-sin a, cos a), so facing (dx, dy) means a = atan2(-dx, dy).
    glm::vec2 toGoal = level.goal.position - level.spawn;
    level.spawnAngle = std::atan2(-toGoal.x, toGoal.y) + rng.uniform(-params.spawnAngleJitter, params.spawnAngleJitter);

    // Obstacles, by rejection sampling: propose one at random, keep it only if it breaks no rule
    int count = rng.uniformInt(params.minObstacles, params.maxObstacles);
    glm::vec2 corridorDir = glm::normalize(toGoal);
    glm::vec2 corridorPerp{-corridorDir.y, corridorDir.x};

    // Optionally a "blocker" first: a planet across the straight line from spawn to goal.
    // Only draws random numbers when enabled, so levels with blockerChance 0 are unchanged.
    if (params.blockerChance > 0.0f && rng.chance(params.blockerChance)) {
        for (int attempt = 0; attempt < ATTEMPTS_PER_OBSTACLE; ++attempt) {
            Obstacle o;
            o.radius = rng.uniform(std::max(params.minRadius, BLOCKER_MIN_RADIUS), std::max(params.maxRadius, BLOCKER_MIN_RADIUS));
            o.density = rng.chance(params.gravityChance) ? rng.uniform(params.minDensity, params.maxDensity) : 0.0f;
            float along = rng.uniform(0.3f, 0.7f);
            float across = rng.uniform(-0.5f, 0.5f) * o.radius;   // centre close enough to the line to block it
            o.position = level.spawn + toGoal * along + corridorPerp * across;
            if (obstacleFits(o, level, params)) {
                level.obstacles.push_back(o);
                break;
            }
        }
        count = std::max(count, (int)level.obstacles.size());
    }
    for (int attempt = 0; attempt < count * ATTEMPTS_PER_OBSTACLE && (int)level.obstacles.size() < count; ++attempt) {
        Obstacle o;
        o.radius = rng.uniform(params.minRadius, params.maxRadius);
        o.density = rng.chance(params.gravityChance) ? rng.uniform(params.minDensity, params.maxDensity) : 0.0f;
        if (rng.chance(params.corridorFraction)) {
            float along = rng.uniform(0.2f, 0.8f);
            float across = rng.uniform(-params.corridorWidth, params.corridorWidth);
            o.position = level.spawn + toGoal * along + corridorPerp * across;
        } else {
            o.position = {rng.uniform(-W, W), rng.uniform(-H, H)};
        }
        if (obstacleFits(o, level, params)) {
            level.obstacles.push_back(o);
        }
    }
    return level;
}
