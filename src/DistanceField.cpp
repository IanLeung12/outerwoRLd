#include "DistanceField.h"
#include "constants.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <utility>

namespace {

constexpr float UNREACHABLE = std::numeric_limits<float>::infinity();

}  // namespace

glm::vec2 DistanceField::cellCentre(int col, int row) const {
    return {-constants::WORLD_HALF_WIDTH + (col + 0.5f) * cellSize.x,
            -constants::WORLD_HALF_HEIGHT + (row + 0.5f) * cellSize.y};
}

void DistanceField::build(const Level& level) {
    const float W = constants::WORLD_HALF_WIDTH;
    const float H = constants::WORLD_HALF_HEIGHT;
    const float shipR = constants::SHIP_RADIUS;

    // Whole cells that cover the world exactly (the cells end up very slightly under CELL_SIZE wide)
    cols = (int)std::round(2.0f * W / CELL_SIZE);
    rows = (int)std::round(2.0f * H / CELL_SIZE);
    cellSize = {2.0f * W / cols, 2.0f * H / rows};
    dist.assign(cols * rows, UNREACHABLE);

    // Blocked cells: anywhere the ship's centre would mean a crash
    std::vector<unsigned char> blocked(cols * rows, 0);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            glm::vec2 c = cellCentre(col, row);
            if (std::abs(c.x) > W - shipR || std::abs(c.y) > H - shipR) {
                blocked[index(col, row)] = 1;
            }
        }
    }
    for (const auto& o : level.obstacles) {
        float reach = o.radius + shipR;
        // Only the cells inside the obstacle's bounding box need checking
        int colLo = std::max(0, (int)std::floor((o.position.x - reach + W) / cellSize.x));
        int colHi = std::min(cols - 1, (int)std::floor((o.position.x + reach + W) / cellSize.x));
        int rowLo = std::max(0, (int)std::floor((o.position.y - reach + H) / cellSize.y));
        int rowHi = std::min(rows - 1, (int)std::floor((o.position.y + reach + H) / cellSize.y));
        for (int row = rowLo; row <= rowHi; ++row) {
            for (int col = colLo; col <= colHi; ++col) {
                if (glm::distance(cellCentre(col, row), o.position) < reach) {
                    blocked[index(col, row)] = 1;
                }
            }
        }
    }

    // Goal cells start at distance 0: anywhere the ship would touch the goal
    using Entry = std::pair<float, int>;   // (distance, cell index). Ties go to the lower index, so it's deterministic.
    std::vector<Entry> storage;
    storage.reserve(cols * rows * 2);   // roomy enough that the queue rarely has to grow
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue(std::greater<Entry>(), std::move(storage));
    glm::vec2 goalMin = level.goal.position - level.goal.halfSize - shipR;
    glm::vec2 goalMax = level.goal.position + level.goal.halfSize + shipR;
    int colLo = std::max(0, (int)std::floor((goalMin.x + W) / cellSize.x));
    int colHi = std::min(cols - 1, (int)std::floor((goalMax.x + W) / cellSize.x));
    int rowLo = std::max(0, (int)std::floor((goalMin.y + H) / cellSize.y));
    int rowHi = std::min(rows - 1, (int)std::floor((goalMax.y + H) / cellSize.y));
    for (int row = rowLo; row <= rowHi; ++row) {
        for (int col = colLo; col <= colHi; ++col) {
            glm::vec2 c = cellCentre(col, row);
            if (c.x >= goalMin.x && c.x <= goalMax.x && c.y >= goalMin.y && c.y <= goalMax.y) {
                dist[index(col, row)] = 0.0f;
                queue.push({0.0f, index(col, row)});
            }
        }
    }

    // Dijkstra: always expand the closest unfinished cell, offering each neighbour a shorter route
    const float diagonal = glm::length(cellSize);
    const int dCol[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    const int dRow[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    const float cost[8] = {cellSize.x, cellSize.x, cellSize.y, cellSize.y, diagonal, diagonal, diagonal, diagonal};
    while (!queue.empty()) {
        auto [d, i] = queue.top();
        queue.pop();
        if (d > dist[i]) continue;   // an outdated entry: this cell was already reached by a shorter route
        int col = i % cols;
        int row = i / cols;
        for (int n = 0; n < 8; ++n) {
            int nc = col + dCol[n];
            int nr = row + dRow[n];
            if (nc < 0 || nc >= cols || nr < 0 || nr >= rows) continue;
            int ni = index(nc, nr);
            if (blocked[ni]) continue;
            float nd = d + cost[n];
            if (nd < dist[ni]) {
                dist[ni] = nd;
                queue.push({nd, ni});
            }
        }
    }
}

float DistanceField::distanceAt(glm::vec2 pos) const {
    // Position in grid units, measured between cell centres
    float gx = (pos.x + constants::WORLD_HALF_WIDTH) / cellSize.x - 0.5f;
    float gy = (pos.y + constants::WORLD_HALF_HEIGHT) / cellSize.y - 0.5f;
    int col0 = (int)std::floor(gx);
    int row0 = (int)std::floor(gy);
    float fx = gx - col0;
    float fy = gy - row0;

    // Bilinear interpolation of the 4 surrounding cells, skipping any that are blocked or unreachable
    float total = 0.0f;
    float weightSum = 0.0f;
    for (int dr = 0; dr <= 1; ++dr) {
        for (int dc = 0; dc <= 1; ++dc) {
            int col = col0 + dc;
            int row = row0 + dr;
            if (col < 0 || col >= cols || row < 0 || row >= rows) continue;
            float d = dist[index(col, row)];
            if (d == UNREACHABLE) continue;
            float weight = (dc ? fx : 1.0f - fx) * (dr ? fy : 1.0f - fy);
            total += weight * d;
            weightSum += weight;
        }
    }
    if (weightSum <= 1e-6f) return -1.0f;
    return total / weightSum;
}
