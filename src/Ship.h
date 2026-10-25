#pragma once
#include <glm/glm.hpp>

struct ShipInput {
    bool thrust = false;
    bool turnLeft = false;
    bool turnRight = false;
};

// The agent's actions are numbered 0-5: {no thrust, thrust} x {no turn, left, right}.
// (Left + right together would cancel out, so that combination isn't an action.)
constexpr int NUM_ACTIONS = 6;

inline ShipInput decodeAction(int action) {
    ShipInput input;
    input.thrust = action / 3 == 1;
    input.turnLeft = action % 3 == 1;
    input.turnRight = action % 3 == 2;
    return input;
}

inline int encodeAction(const ShipInput& input) {
    int turn = input.turnLeft == input.turnRight ? 0 : (input.turnLeft ? 1 : 2);
    return (input.thrust ? 3 : 0) + turn;
}

class Ship {
public:
    void update(float dt, const ShipInput& input, glm::vec2 forces);
    glm::mat4 modelMatrix() const;
    void reset();
    glm::vec2 getPosition() const;
    glm::vec2 getVelocity() const;
    float getAngle() const;
    void setSpawn(glm::vec2 point, float angle);

private:
    glm::vec2 position{0.0f, 0.0f};
    glm::vec2 velocity{0.0f, 0.0f};
    float angle = 0.0f;
    glm::vec2 spawnPoint{0.0f, 0.0f};
    float spawnAngle = 0.0f;
};
