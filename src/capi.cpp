#include "capi.h"
#include "LevelGen.h"
#include "World.h"
#include <algorithm>
#include <cmath>
#include <vector>

struct OwWorld {
    World world{makeTestLevel()};
};

struct OwBatch {
    std::vector<World> worlds;
};

static int resultCode(StepResult result) {
    switch (result) {
        case StepResult::Running: return 0;
        case StepResult::Crashed: return 1;
        case StepResult::ReachedGoal: return 2;
    }
    return 0;
}

static float shapingDistance(const World& world, int shaping) {
    if (shaping == 0) return world.goalDistance();
    if (shaping == 1) return world.pathDistance();
    return 0.0f;
}

int ow_obs_size(void) {
    return constants::OBS_SIZE;
}

int ow_num_actions(void) {
    return NUM_ACTIONS;
}

OwWorld* ow_create(void) {
    return new OwWorld();
}

void ow_destroy(OwWorld* w) {
    delete w;
}

void ow_load_test_level(OwWorld* w) {
    w->world.load(makeTestLevel());
}

void ow_load_generated(OwWorld* w, unsigned int seed, float difficulty) {
    w->world.load(generateLevel(seed, paramsForDifficulty(difficulty)));
}

void ow_reset(OwWorld* w) {
    w->world.reset();
}

int ow_step(OwWorld* w, int action) {
    return resultCode(w->world.step(decodeAction(action)));
}

void ow_observe(const OwWorld* w, float* out) {
    Observation obs = w->world.observe();
    std::copy(obs.begin(), obs.end(), out);
}

float ow_goal_distance(const OwWorld* w) {
    return w->world.goalDistance();
}

float ow_path_distance(const OwWorld* w) {
    return w->world.pathDistance();
}

int ow_step_count(const OwWorld* w) {
    return w->world.getStepCount();
}

OwBatch* ow_batch_create(int num_worlds) {
    OwBatch* batch = new OwBatch();
    batch->worlds.reserve(num_worlds);
    for (int i = 0; i < num_worlds; ++i) {
        batch->worlds.emplace_back(makeTestLevel());
    }
    return batch;
}

void ow_batch_destroy(OwBatch* batch) {
    delete batch;
}

void ow_batch_load_test_level(OwBatch* batch, int index) {
    batch->worlds[index].load(makeTestLevel());
}

void ow_batch_load_generated(OwBatch* batch, int index, unsigned int seed, float difficulty) {
    batch->worlds[index].load(generateLevel(seed, paramsForDifficulty(difficulty)));
}

void ow_batch_load_generated_ex(OwBatch* batch, int index, unsigned int seed, float difficulty,
                                float blocker_chance) {
    GenParams params = paramsForDifficulty(difficulty);
    params.blockerChance = blocker_chance;
    batch->worlds[index].load(generateLevel(seed, params));
}

void ow_batch_step(OwBatch* batch, const int* actions, int shaping,
                   int* results, int* step_counts, float* distances, float* observations) {
    ow_batch_step_repeat(batch, actions, 1, shaping, results, step_counts, distances, observations);
}

void ow_batch_step_repeat(OwBatch* batch, const int* actions, int repeat, int shaping,
                          int* results, int* step_counts, float* distances, float* observations) {
    for (size_t i = 0; i < batch->worlds.size(); ++i) {
        World& world = batch->worlds[i];
        ShipInput input = decodeAction(actions[i]);
        StepResult result = StepResult::Running;
        for (int r = 0; r < repeat && result == StepResult::Running; ++r) {
            result = world.step(input);
        }
        results[i] = resultCode(result);
        step_counts[i] = world.getStepCount();
        distances[i] = shapingDistance(world, shaping);
        Observation obs = world.observe();
        std::copy(obs.begin(), obs.end(), observations + i * constants::OBS_SIZE);
    }
}

// How far the ship's centre can move from origin along the unit vector dir before it crashes.
static float rayDistance(const Level& level, glm::vec2 origin, glm::vec2 dir, float maxRange) {
    const float shipR = constants::SHIP_RADIUS;
    float best = maxRange;
    // World edges, moved in by the ship's radius (where outOfBounds() triggers)
    glm::vec2 limit{constants::WORLD_HALF_WIDTH - shipR, constants::WORLD_HALF_HEIGHT - shipR};
    for (int axis = 0; axis < 2; ++axis) {
        if (dir[axis] > 1e-6f) best = std::min(best, (limit[axis] - origin[axis]) / dir[axis]);
        if (dir[axis] < -1e-6f) best = std::min(best, (-limit[axis] - origin[axis]) / dir[axis]);
    }
    // Planets, grown by the ship's radius: solve |origin + t*dir - centre| = r for the first t >= 0
    for (const auto& o : level.obstacles) {
        glm::vec2 c = o.position - origin;
        float r = o.radius + shipR;
        float b = glm::dot(c, dir);
        float disc = b * b - (glm::dot(c, c) - r * r);
        if (disc < 0.0f) continue;   // the ray misses
        float root = std::sqrt(disc);
        if (b + root < 0.0f) continue;   // entirely behind
        best = std::min(best, std::max(0.0f, b - root));   // 0 if already touching
    }
    return std::max(0.0f, best);
}

void ow_batch_rays(const OwBatch* batch, int num_rays, float max_range, float* out) {
    for (size_t i = 0; i < batch->worlds.size(); ++i) {
        const World& world = batch->worlds[i];
        float a = world.getShip().getAngle();
        glm::vec2 right{std::cos(a), std::sin(a)};
        glm::vec2 forward{-std::sin(a), std::cos(a)};
        for (int k = 0; k < num_rays; ++k) {
            float theta = 2.0f * 3.14159265f * (float)k / (float)num_rays;
            glm::vec2 dir = forward * std::cos(theta) + right * std::sin(theta);
            out[i * num_rays + k] = rayDistance(world.getLevel(), world.getShip().getPosition(), dir, max_range) / max_range;
        }
    }
}

void ow_batch_observe_one(const OwBatch* batch, int index, float* out) {
    Observation obs = batch->worlds[index].observe();
    std::copy(obs.begin(), obs.end(), out);
}

float ow_batch_distance_one(const OwBatch* batch, int index, int shaping) {
    return shapingDistance(batch->worlds[index], shaping);
}
