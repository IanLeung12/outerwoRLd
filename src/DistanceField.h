#pragma once
#include <vector>
#include <glm/glm.hpp>
#include "Level.h"

// For every point in a level, how far the ship has to travel to reach the goal, going around
// obstacles instead of through them. Used for the progress reward, so flying around a planet
// counts as getting closer.
//
// The world is split into a grid. Cells the ship can't be in (inside an obstacle or past the
// edge, allowing for the ship's radius) are blocked, and the distance from the goal spreads out
// to every other cell with Dijkstra's algorithm, moving to the 8 neighbouring cells.
class DistanceField {
public:
    static constexpr float CELL_SIZE = 0.02f;

    void build(const Level& level);

    // Travel distance from pos to the goal, interpolated between the 4 nearest cells.
    // Negative if none of those cells can reach the goal (for example deep inside an obstacle).
    float distanceAt(glm::vec2 pos) const;

    bool reachable(glm::vec2 pos) const { return distanceAt(pos) >= 0.0f; }

    const std::vector<float>& cells() const { return dist; }

private:
    int index(int col, int row) const { return row * cols + col; }
    glm::vec2 cellCentre(int col, int row) const;

    int cols = 0;
    int rows = 0;
    glm::vec2 cellSize{CELL_SIZE, CELL_SIZE};
    std::vector<float> dist;   // UNREACHABLE for blocked cells and cells with no path to the goal
};
