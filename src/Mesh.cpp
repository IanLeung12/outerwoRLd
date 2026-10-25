#include "Mesh.h"
#include <glad/glad.h>
#include <numbers>
#include <cmath>

Mesh::Mesh(const std::vector<float>& vertices) {
    vertexCount = vertices.size()/3;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
}

Mesh::~Mesh() {
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
}

void Mesh::draw(unsigned int mode) const {
    glBindVertexArray(vao);
    glDrawArrays(mode, 0, vertexCount);
}

std::vector<float> Mesh::makeCircle(float radius, int segments) {
    std::vector<float> circle;
    for (int i = 0; i < segments; ++i) {
        float angle  = (float)i / segments * 2 * std::numbers::pi_v<float>;
        circle.insert(circle.end(), {std::cos(angle) * radius, std::sin(angle) * radius, 0.0f});
    }
    return circle;
}

std::vector<float> Mesh::makeRect() {
    return std::vector<float> {-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0};
}