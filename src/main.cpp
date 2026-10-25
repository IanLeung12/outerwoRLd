
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "constants.h"
#include "Shader.h"
#include "Mesh.h"
#include "Ship.h"
#include "Obstacle.h"
#include "collision.h"
#include "Goal.h"
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

        Mesh circleMesh(Mesh::makeCircle(1.0f, 12));

        Mesh goalMesh(Mesh::makeRect());

        std::vector<Obstacle> obstacles = {
            {{ 0.8f,  0.4f}, 0.15f},
            {{-0.9f, -0.5f}, 0.25f},
            {{-0.6f,  0.6f}, 0.10f},
            {{ 1.2f, -0.3f}, 0.20f},
        };

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
            uniform vec4 uColor;
            out vec4 FragColor;

            void main() {
                FragColor = uColor;
            }
            
        )";

    
        Shader shader(vertexSrc, fragSrc);

        Ship ship;
        ship.setSpawn({-1.5f, 0.0f});
        ship.reset();
        
        Goal goal{{1.6f, 0.0f}, {0.08f, 0.3f}};
        double lastTime = glfwGetTime();
        while(!glfwWindowShouldClose(window)) {
            int w, h;
            glfwGetFramebufferSize(window, &w, &h);
            glViewport(0,0,w,h);
            float aspect = (float)w / (float)h;
            glm::mat4 projection = glm::ortho(-aspect, aspect, -1.0f, 1.0f, -1.0f, 1.0f);

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

            if (std::abs(ship.getPosition().x) > aspect - constants::SHIP_RADIUS) {
                ship.reset();
            } else if (std::abs(ship.getPosition().y) > 1.0f - constants::SHIP_RADIUS) {
                ship.reset();
            }
            
            for (const auto& obstacle: obstacles) {
                if (circlesOverlap(ship.getPosition(), constants::SHIP_RADIUS, obstacle.position, obstacle.radius)) {
                    ship.reset();
                    break;
                }
            }
            
            if (circleRectOverlap(ship.getPosition(), constants::SHIP_RADIUS, goal.position, goal.halfSize)) {
                std::cout << "Level complete!\n";
                ship.reset();
            }

            glm::mat4 model(1.0f);
            model = ship.modelMatrix();

            glClear(GL_COLOR_BUFFER_BIT);
            shader.use();
            shader.setVec4("uColor", constants::COLOR_SHIP);
            shader.setMat4("uTransform", projection * model);
            shipMesh.draw(GL_LINE_LOOP);
            
            shader.setVec4("uColor", constants::COLOR_OBSTACLE);
            for (const Obstacle& o : obstacles) {
                shader.setMat4("uTransform", projection * o.modelMatrix());
                circleMesh.draw(GL_LINE_LOOP);
            }
            
            shader.setVec4("uColor", constants::COLOR_GOAL);
            shader.setMat4("uTransform", projection * goal.modelMatrix());
            goalMesh.draw(GL_LINE_LOOP);

            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }
    glfwTerminate();
    return 0;
}