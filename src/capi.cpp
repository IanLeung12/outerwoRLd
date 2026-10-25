#include "capi.h"
#include "World.h"
#include <algorithm>

struct OwWorld {
    World world{makeTestLevel()};
};

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

void ow_reset(OwWorld* w) {
    w->world.reset();
}

int ow_step(OwWorld* w, int action) {
    StepResult result = w->world.step(decodeAction(action));
    switch (result) {
        case StepResult::Running: return 0;
        case StepResult::Crashed: return 1;
        case StepResult::ReachedGoal: return 2;
    }
    return 0;
}

void ow_observe(const OwWorld* w, float* out) {
    Observation obs = w->world.observe();
    std::copy(obs.begin(), obs.end(), out);
}

float ow_goal_distance(const OwWorld* w) {
    return w->world.goalDistance();
}

int ow_step_count(const OwWorld* w) {
    return w->world.getStepCount();
}
