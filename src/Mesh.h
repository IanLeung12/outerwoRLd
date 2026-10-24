#pragma once
#include <vector>

class Mesh {
public:
    Mesh(const std::vector<float>& vertices);
    ~Mesh();

    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

    void draw(unsigned int mode) const;

private:
    unsigned int vao;
    unsigned int vbo;
    int vertexCount;
};
