
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "constants.h"
#include "Shader.h"
#include <cmath>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

int main() {
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
    GLFWwindow* window = glfwCreateWindow(constants::WINDOW_WIDTH, constants::WINDOW_HEIGHT, constants::WINDOW_TITLE, nullptr, nullptr);
    if (!window) {glfwTerminate(); return -1;}
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {glfwTerminate(); return -1;}
    glClearColor(0,0,0,1);

    unsigned int vao;
    unsigned int vbo;
    float vertices[3][3] = {{0.03,-0.05,0}, {0,0.05,0}, {-0.03,-0.05,0}};
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    const char* vertexSrc = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        uniform mat4 uTransform;

        void main() {
            gl_Position = uTransform * vec4(aPos, 1.0);

        }
    )";

    const char* fragSrc = R"(
        #version 330 core
        out vec4 FragColor;

        void main() {
            FragColor = vec4(1.0,0.4,0.4,1.0);
        }
        
    )";

    {
        Shader shader(vertexSrc, fragSrc);

        double x = 0.0;
        double y = 0.0;
        double vx = 0.0;
        double vy = 0.0;
        float angle = 0.0f;
        double lastTime = glfwGetTime();
        while(!glfwWindowShouldClose(window)) {
            double now = glfwGetTime();
            double dt = now - lastTime;
            lastTime = now;
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
                glm::vec2 forward = glm::vec2(-sin(angle), cos(angle));
                vx += forward.x * constants::SHIP_THRUST * dt;
                vy += forward.y * constants::SHIP_THRUST * dt;
            }
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
                angle += constants::SHIP_TURN_SPEED * dt;
            }
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
                angle -= constants::SHIP_TURN_SPEED * dt;
            }
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
                vx = 0;
                vy = 0;
            }
            double dragFactor = std::pow(constants::SHIP_DRAG, dt);
            vx *= dragFactor;
            vy *= dragFactor;
            x += vx * dt;
            y += vy * dt;
            
            int w, h;
            glfwGetFramebufferSize(window, &w, &h);
            glViewport(0,0,w,h);
            float aspect = (float)w / (float)h;
            glm::mat4 projection = glm::ortho(-aspect, aspect, -1.0f, 1.0f, -1.0f, 1.0f);

            glm::mat4 model(1.0f);
            model = glm::translate(model, glm::vec3(x, y, 0.0));
            model = glm::rotate(model, angle, glm::vec3(0,0,1));

            glClear(GL_COLOR_BUFFER_BIT);
            shader.use();
            shader.setMat4("uTransform", projection * model);
            glBindVertexArray(vao);
            glDrawArrays(GL_LINE_LOOP, 0, 3);
            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }
    glfwTerminate();
    return 0;
}