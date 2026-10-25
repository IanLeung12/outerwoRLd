#pragma once
#include <glm/glm.hpp>

namespace constants {
    // Window
    constexpr int WINDOW_WIDTH = 1920;
    constexpr int WINDOW_HEIGHT = 1080;
    constexpr const char* WINDOW_TITLE = "Outerworld";

    // Ship
    constexpr float SHIP_MOVE_SPEED = 0.5f;   // units per second (direct movement, for now)
    constexpr float SHIP_TURN_SPEED = 3.5f;   // radians per second
    constexpr float SHIP_THRUST = 2.0f;       // units per second, per second
    constexpr float SHIP_DRAG = 0.75f;      
    constexpr float SHIP_RADIUS = 0.04f;   // hitbox

    // Obstacles
    constexpr float PLANET_DENSITY = 0.2f;
    constexpr float GRAVITY_RANGE = 3.0f;
    

    // Colors
    constexpr glm::vec4 COLOR_SHIP{1.0f, 1.0f, 1.0f, 1.0f};
    constexpr glm::vec4 COLOR_OBSTACLE{1.0f, 0.2f, 0.2f, 1.0f};
    constexpr glm::vec4 COLOR_GOAL{0.3f, 1.0f, 0.3f, 1.0f};
    constexpr glm::vec4 COLOR_GRAVITY{0.2f, 0.2f, 0.5f, 0.5f};
}
