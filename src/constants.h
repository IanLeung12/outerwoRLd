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
    constexpr float SHIP_THRUST = 1.5f;       // units per second, per second
    constexpr float SHIP_DRAG = 0.75f;      
    constexpr float SHIP_RADIUS = 0.04f;   // hitbox

    constexpr glm::vec4 COLOR_SHIP{1.0f, 1.0f, 1.0f, 1.0f};
    constexpr glm::vec4 COLOR_OBSTACLE{1.0f, 0.2f, 0.2f, 1.0f};
    constexpr glm::vec4 COLOR_GOAL{0.3f, 1.0f, 0.3f, 1.0f};
}
