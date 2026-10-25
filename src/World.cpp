#include "World.h"
#include "collision.h"
#include "constants.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

World::World(const Level& level) {
    load(level);
}

void World::load(const Level& newLevel) {
    level = newLevel;
    reset();
}

void World::reset() {
    ship.setSpawn(level.spawn, level.spawnAngle);
    ship.reset();
    stepCount = 0;
}

StepResult World::step(const ShipInput& input) {
    glm::vec2 gravity = Obstacle::gravityAt(ship.getPosition(), level.obstacles);
    ship.update(constants::SIM_DT, input, gravity);
    ++stepCount;

    if (outOfBounds() || hitObstacle()) {
        return StepResult::Crashed;
    }
    if (reachedGoal()) {
        return StepResult::ReachedGoal;
    }
    return StepResult::Running;
}

bool World::outOfBounds() const {
    glm::vec2 pos = ship.getPosition();
    return std::abs(pos.x) > constants::WORLD_HALF_WIDTH - constants::SHIP_RADIUS
        || std::abs(pos.y) > constants::WORLD_HALF_HEIGHT - constants::SHIP_RADIUS;
}

bool World::hitObstacle() const {
    for (const auto& obstacle : level.obstacles) {
        if (circlesOverlap(ship.getPosition(), constants::SHIP_RADIUS, obstacle.position, obstacle.radius)) {
            return true;
        }
    }
    return false;
}

bool World::reachedGoal() const {
    return circleRectOverlap(ship.getPosition(), constants::SHIP_RADIUS, level.goal.position, level.goal.halfSize);
}

float World::goalDistance() const {
    glm::vec2 pos = ship.getPosition();
    glm::vec2 closest = glm::clamp(pos, level.goal.position - level.goal.halfSize, level.goal.position + level.goal.halfSize);
    return glm::distance(pos, closest);
}

// Rotates a world-space vector into the ship's frame: x = to the ship's right, y = forward.
glm::vec2 World::toShipFrame(glm::vec2 v) const {
    float a = ship.getAngle();
    glm::vec2 right{std::cos(a), std::sin(a)};
    glm::vec2 forward{-std::sin(a), std::cos(a)};
    return {glm::dot(v, right), glm::dot(v, forward)};
}

Observation World::observe() const {
    Observation obs{};   // all zeros, so unused obstacle slots stay empty
    int i = 0;
    auto put = [&](float value) { obs[i++] = value; };

    glm::vec2 pos = ship.getPosition();

    glm::vec2 vel = toShipFrame(ship.getVelocity()) / constants::OBS_VELOCITY_SCALE;
    put(vel.x);
    put(vel.y);

    put(std::sin(ship.getAngle()));
    put(std::cos(ship.getAngle()));

    glm::vec2 goal = toShipFrame(level.goal.position - pos) / constants::OBS_POSITION_SCALE;
    put(goal.x);
    put(goal.y);
    put(level.goal.halfSize.x / constants::OBS_POSITION_SCALE);
    put(level.goal.halfSize.y / constants::OBS_POSITION_SCALE);

    put((pos.x + constants::WORLD_HALF_WIDTH) / (2.0f * constants::WORLD_HALF_WIDTH));
    put((constants::WORLD_HALF_WIDTH - pos.x) / (2.0f * constants::WORLD_HALF_WIDTH));
    put((pos.y + constants::WORLD_HALF_HEIGHT) / (2.0f * constants::WORLD_HALF_HEIGHT));
    put((constants::WORLD_HALF_HEIGHT - pos.y) / (2.0f * constants::WORLD_HALF_HEIGHT));

    glm::vec2 gravity = toShipFrame(Obstacle::gravityAt(pos, level.obstacles)) / constants::SHIP_THRUST;
    put(gravity.x);
    put(gravity.y);

    assert(i == constants::OBS_SIZE - constants::OBS_MAX_OBSTACLES * constants::OBS_VALUES_PER_OBSTACLE);

    // Sort by distance to the obstacle's surface, so slot 1 is always the most immediate danger.
    std::vector<const Obstacle*> nearest;
    for (const auto& o : level.obstacles) {
        nearest.push_back(&o);
    }
    auto surfaceDistance = [&](const Obstacle* o) { return glm::distance(pos, o->position) - o->radius; };
    std::sort(nearest.begin(), nearest.end(), [&](const Obstacle* a, const Obstacle* b) {
        return surfaceDistance(a) < surfaceDistance(b);
    });

    int count = std::min((int)nearest.size(), constants::OBS_MAX_OBSTACLES);
    for (int n = 0; n < count; ++n) {
        const Obstacle* o = nearest[n];
        glm::vec2 rel = toShipFrame(o->position - pos) / constants::OBS_POSITION_SCALE;
        put(rel.x);
        put(rel.y);
        put(o->radius / constants::OBS_RADIUS_SCALE);
        put(o->density / constants::PLANET_DENSITY);
        put(1.0f);
    }

    assert(i <= constants::OBS_SIZE);
    return obs;
}
