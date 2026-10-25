#pragma once
#include <glm/glm.hpp>

namespace constants {
    // Window
    constexpr int WINDOW_WIDTH = 1920;
    constexpr int WINDOW_HEIGHT = 1080;
    constexpr const char* WINDOW_TITLE = "Outerworld";

    // Simulation
    constexpr float SIM_DT = 1.0f / 60.0f;          // fixed timestep: 60 steps per simulated second
    constexpr float WORLD_HALF_WIDTH = 16.0f / 9.0f; // the level is a fixed 16:9 area, independent of the window
    constexpr float WORLD_HALF_HEIGHT = 1.0f;

    // Ship
    constexpr float SHIP_MOVE_SPEED = 0.5f;   // units per second (direct movement, for now)
    constexpr float SHIP_TURN_SPEED = 3.5f;   // radians per second
    constexpr float SHIP_THRUST = 2.0f;       // units per second, per second
    constexpr float SHIP_DRAG = 0.75f;      
    constexpr float SHIP_RADIUS = 0.04f;   // hitbox

    // Obstacles
    constexpr float PLANET_DENSITY = 12.0f;
    constexpr float GRAVITY_RANGE = 3.0f;

    // Observation (what the RL agent sees). Scales bring values to roughly -1..1.
    constexpr int OBS_MAX_OBSTACLES = 8;          // K nearest obstacles; empty slots are zeros
    constexpr int OBS_VALUES_PER_OBSTACLE = 5;    // relative x, relative y, radius, density, present
    constexpr int OBS_SIZE = 14 + OBS_MAX_OBSTACLES * OBS_VALUES_PER_OBSTACLE;
    constexpr float OBS_POSITION_SCALE = WORLD_HALF_WIDTH;
    constexpr float OBS_VELOCITY_SCALE = 3.0f;
    constexpr float OBS_RADIUS_SCALE = 0.25f;

    // Colors
    constexpr glm::vec4 COLOR_SHIP{1.0f, 1.0f, 1.0f, 1.0f};
    constexpr glm::vec4 COLOR_OBSTACLE{1.0f, 0.2f, 0.2f, 1.0f};
    constexpr glm::vec4 COLOR_GOAL{0.3f, 1.0f, 0.3f, 1.0f};
    constexpr glm::vec4 COLOR_GRAVITY{0.2f, 0.2f, 0.5f, 0.5f};
    constexpr glm::vec4 COLOR_BOUNDS{0.4f, 0.4f, 0.4f, 1.0f};
}
