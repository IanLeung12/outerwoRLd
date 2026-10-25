#pragma once
#include <array>
#include "constants.h"
#include "Level.h"
#include "Ship.h"

// What the agent sees, as a fixed-length list of numbers. Everything is relative to the ship
// and rotated into its frame (y = forward, x = right), so the same situation looks the same
// in any level. Layout:
//   [0-1]   velocity (right, forward)
//   [2-3]   angle as sin, cos
//   [4-5]   goal centre relative to ship (right, forward)
//   [6-7]   goal half size (x, y)
//   [8-11]  distance to the left, right, bottom, top boundary (0..1)
//   [12-13] gravity pull on the ship (right, forward), in units of engine thrust
//   [14-]   OBS_MAX_OBSTACLES obstacles, nearest surface first, each:
//           relative right, relative forward, radius, density, present (1 = real, 0 = empty slot)
using Observation = std::array<float, constants::OBS_SIZE>;

enum class StepResult {
    Running,
    Crashed,
    ReachedGoal,
};

// The game simulation, with no window or OpenGL. Advances in fixed steps of constants::SIM_DT,
// so the same level and the same inputs always produce the same result.
class World {
public:
    explicit World(const Level& level);

    void load(const Level& level);   // switch to a new level and respawn
    void reset();                    // respawn the ship in the current level

    StepResult step(const ShipInput& input);
    Observation observe() const;
    float goalDistance() const;   // straight-line distance from the ship to the nearest point of the goal

    const Ship& getShip() const { return ship; }
    const Level& getLevel() const { return level; }
    int getStepCount() const { return stepCount; }

private:
    bool outOfBounds() const;
    bool hitObstacle() const;
    bool reachedGoal() const;
    glm::vec2 toShipFrame(glm::vec2 v) const;

    Level level;
    Ship ship;
    int stepCount = 0;
};
