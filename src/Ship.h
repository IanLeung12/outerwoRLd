#pragma once
#include <glm/glm.hpp>

struct ShipInput {
    bool thrust = false;
    bool turnLeft = false;
    bool turnRight = false;
};

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
