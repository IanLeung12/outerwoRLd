#pragma once
#include <vector>

class Mesh {
public:
    Mesh(const std::vector<float>& vertices);
    ~Mesh();

    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

    void draw(unsigned int mode) const;

    static std::vector<float> makeCircle(float radius, int segments);
    static std::vector<float> makeRect();

private:
    unsigned int vao;
    unsigned int vbo;
    int vertexCount;
};
