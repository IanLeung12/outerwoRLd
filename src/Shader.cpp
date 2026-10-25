#include "Shader.h"
#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>

static unsigned int compileShader(GLenum type, const char* src) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    int success;
    char log[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, nullptr, log);
        const char* name = (type == GL_VERTEX_SHADER) ? "Vertex" : "Fragment";
        std::cerr << name << " shader error:\n" << log << "\n";
    }
    return shader;
}

Shader::Shader(const char* vertexSrc, const char* fragSrc) {
    unsigned int vs = compileShader(GL_VERTEX_SHADER, vertexSrc);
    unsigned int fs = compileShader(GL_FRAGMENT_SHADER, fragSrc);

    id = glCreateProgram();
    glAttachShader(id, vs);
    glAttachShader(id, fs);
    glLinkProgram(id);
    glDeleteShader(vs);
    glDeleteShader(fs);
    int success;
    char log[512];
    glGetProgramiv(id, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(id, 512, nullptr, log);
        std::cerr << "Link error:\n" << log << "\n";
    }

}

Shader::~Shader() {
    glDeleteProgram(id);
}

void Shader::use() const {
    glUseProgram(id);
}

void Shader::setMat4(const char* name, const glm::mat4& value) const {
    int transformLoc = glGetUniformLocation(id, name);
    glUniformMatrix4fv(transformLoc, 1, GL_FALSE, glm::value_ptr(value));
}

void Shader::setVec4(const char* name, const glm::vec4& value) const {
    int loc = glGetUniformLocation(id, name);
    glUniform4f(loc, value.r, value.g, value.b, value.a);
}
