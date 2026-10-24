#include "Mesh.h"
#include <glad/glad.h>

Mesh::Mesh(const std::vector<float>& vertices) {
    // TODO: create the VAO and VBO, upload the vertices, set up the attribute, set vertexCount
}

Mesh::~Mesh() {
    // TODO: delete the VAO and VBO
}

void Mesh::draw(unsigned int mode) const {
    // TODO: bind the VAO and draw vertexCount vertices
}
