
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "constants.h"
#include "Shader.h"
#include "Mesh.h"
#include "Ship.h"
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

    {
        std::vector<float> vertices = {
            0.03f, -0.05f, 0.0f,
            0.0f,   0.05f, 0.0f,
            -0.03f, -0.05f, 0.0f,
        };
        Mesh shipMesh(vertices);

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

    
        Shader shader(vertexSrc, fragSrc);

        Ship ship;
        double lastTime = glfwGetTime();
        while(!glfwWindowShouldClose(window)) {
            double now = glfwGetTime();
            double dt = now - lastTime;
            lastTime = now;
            ShipInput shipinput;
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
                shipinput.thrust = true;
            }
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
                shipinput.turnLeft = true;
            }
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
                shipinput.turnRight = true;
            }
            ship.update(dt, shipinput);
            
            int w, h;
            glfwGetFramebufferSize(window, &w, &h);
            glViewport(0,0,w,h);
            float aspect = (float)w / (float)h;
            glm::mat4 projection = glm::ortho(-aspect, aspect, -1.0f, 1.0f, -1.0f, 1.0f);

            glm::mat4 model(1.0f);
            model = ship.modelMatrix();

            glClear(GL_COLOR_BUFFER_BIT);
            shader.use();
            shader.setMat4("uTransform", projection * model);
            shipMesh.draw(GL_LINE_LOOP);
            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }
    glfwTerminate();
    return 0;
}