#pragma once
#include <glm/glm.hpp>

class Shader {
public:
    Shader(const char* vertexSrc, const char* fragSrc);
    ~Shader();

    Shader(const Shader&) = delete;
    Shader& operator=(const Shader&) = delete;

    void use() const;
    void setMat4(const char* name, const glm::mat4& value) const;
    void setVec4(const char* name, const glm::vec4& value) const;

private:
    unsigned int id;
};
