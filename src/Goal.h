#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct Goal {
    glm::vec2 position;   // centre of the rectangle
    glm::vec2 halfSize;   // distance from the centre to each edge (half the width, half the height)

    glm::mat4 modelMatrix() const {
        glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(position, 0.0f));
        return glm::scale(model, glm::vec3(halfSize, 1.0f));
    }
};
