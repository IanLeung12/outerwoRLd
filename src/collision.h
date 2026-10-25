#pragma once
#include <glm/glm.hpp>

inline bool circlesOverlap(glm::vec2 posA, float radiusA, glm::vec2 posB, float radiusB) {return glm::distance(posA, posB) < (radiusA + radiusB);}

inline bool circleRectOverlap(glm::vec2 circlePos, float radius, glm::vec2 rectPos, glm::vec2 rectHalfSize) {
    glm::vec2 closest = glm::clamp(circlePos, rectPos - rectHalfSize, rectPos + rectHalfSize);
    return glm::distance(closest, circlePos) < radius;
}