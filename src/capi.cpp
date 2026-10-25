#include "capi.h"
#include "LevelGen.h"
#include "World.h"
#include <algorithm>
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

void ow_batch_step(OwBatch* batch, const int* actions, int shaping,
                   int* results, int* step_counts, float* distances, float* observations) {
    for (size_t i = 0; i < batch->worlds.size(); ++i) {
        World& world = batch->worlds[i];
        results[i] = resultCode(world.step(decodeAction(actions[i])));
        step_counts[i] = world.getStepCount();
        distances[i] = shapingDistance(world, shaping);
        Observation obs = world.observe();
        std::copy(obs.begin(), obs.end(), observations + i * constants::OBS_SIZE);
    }
}

void ow_batch_observe_one(const OwBatch* batch, int index, float* out) {
    Observation obs = batch->worlds[index].observe();
    std::copy(obs.begin(), obs.end(), out);
}

float ow_batch_distance_one(const OwBatch* batch, int index, int shaping) {
    return shapingDistance(batch->worlds[index], shaping);
}
