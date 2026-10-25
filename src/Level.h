#pragma once
#include <glm/glm.hpp>
#include <numbers>
#include <vector>
#include "constants.h"
#include "Goal.h"
#include "Obstacle.h"

// Everything that defines one level. A World simulates a Level.
struct Level {
    glm::vec2 spawn{0.0f, 0.0f};
    float spawnAngle = 0.0f;   // radians, 0 = pointing up
    Goal goal;
    std::vector<Obstacle> obstacles;
};

// The hand-made level from before. Will be joined by a seeded generator later.
inline Level makeTestLevel() {
    Level level;
    level.spawn = {-1.5f, 0.0f};
    level.spawnAngle = -0.5f * std::numbers::pi_v<float>;   // facing right
    level.goal = {{1.6f, 0.0f}, {0.08f, 0.3f}};
    level.obstacles = {
        {{ 0.8f,  0.4f}, 0.15f, constants::PLANET_DENSITY},
        {{-0.9f, -0.5f}, 0.25f, constants::PLANET_DENSITY},
        {{-0.6f,  0.6f}, 0.10f, constants::PLANET_DENSITY},
        {{ 1.2f, -0.3f}, 0.20f, constants::PLANET_DENSITY},
    };
    return level;
}
