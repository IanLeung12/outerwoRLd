#include "Ship.h"
#include "constants.h"
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>
#include <numbers>

void Ship::update(float dt, const ShipInput& input, glm::vec2 forces) {
    if (input.thrust) {
        glm::vec2 forward = glm::vec2(-sin(angle), cos(angle));
        velocity.x += forward.x * constants::SHIP_THRUST * dt;
        velocity.y += forward.y * constants::SHIP_THRUST * dt;
    }
    if (input.turnLeft) {
        angle += constants::SHIP_TURN_SPEED * dt;
    }
    if (input.turnRight) {
        angle -= constants::SHIP_TURN_SPEED * dt;
    }
    velocity += forces * dt;
    velocity *= std::pow(constants::SHIP_DRAG, dt);
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;
}

glm::mat4 Ship::modelMatrix() const {
    glm::mat4 model(1.0f);
    model = glm::translate(model, glm::vec3(position.x, position.y, 0.0));
    model = glm::rotate(model, angle, glm::vec3(0,0,1));
    return model;
}

void Ship::reset() {
    position = spawnPoint;
    velocity.x = 0;
    velocity.y = 0;
    angle = spawnAngle;
}

glm::vec2 Ship::getPosition() const {
    return position;
}

glm::vec2 Ship::getVelocity() const {
    return velocity;
}

float Ship::getAngle() const {
    return angle;
}

void Ship::setSpawn(glm::vec2 point, float angle) {
    spawnPoint = point;
    spawnAngle = angle;
}
