#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct Obstacle {
    glm::vec2 position;
    float radius;

    glm::mat4 modelMatrix() const {
        glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(position, 0.0f));
        return glm::scale(model, glm::vec3(radius, radius, 1.0f));
    }
};
