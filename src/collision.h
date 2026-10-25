#pragma once
#include <glm/glm.hpp>

inline bool circlesOverlap(glm::vec2 posA, float radiusA, glm::vec2 posB, float radiusB) {return glm::distance(posA, posB) < (radiusA + radiusB);}