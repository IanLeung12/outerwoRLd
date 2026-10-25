#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include "constants.h"
struct Obstacle {
    glm::vec2 position;
    float radius;
    float density = 0.0f;

    glm::mat4 modelMatrix() const {
        glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(position, 0.0f));
        return glm::scale(model, glm::vec3(radius, radius, 1.0f));
    }

    static glm::vec2 gravityAt(glm::vec2 pos, const std::vector<Obstacle>& obstacles) {
        glm::vec2 total{0.0f, 0.0f};
        for (const auto& o: obstacles) {
            if (o.density > 0) {
                glm::vec2 dir = o.position - pos;
                float dist = glm::length(dir);
                if (dist >= o.gravityRange() || dist == 0) {
                    continue;
                }
                dist = std::max(dist, o.radius);
                float surfaceGravity = o.density * o.radius;
                float strength = surfaceGravity * (o.radius/dist) * (o.radius/dist);
                total += glm::normalize(dir) * strength;
            }
        }
        return total;
    }

    float gravityRange() const { return constants::GRAVITY_RANGE * radius; }
};
