
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "constants.h"
#include "Shader.h"
#include "Mesh.h"
#include "World.h"
#include <algorithm>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Shows the whole world, keeping its 16:9 shape whatever the window size (black bars if needed).
static glm::mat4 fitWorldProjection(int w, int h) {
    float windowAspect = (float)w / (float)h;
    float worldAspect = constants::WORLD_HALF_WIDTH / constants::WORLD_HALF_HEIGHT;
    float halfW = constants::WORLD_HALF_WIDTH;
    float halfH = constants::WORLD_HALF_HEIGHT;
    if (windowAspect > worldAspect) {
        halfW = halfH * windowAspect;   // window is wider than the world
    } else {
        halfH = halfW / windowAspect;   // window is taller than the world
    }
    return glm::ortho(-halfW, halfW, -halfH, halfH, -1.0f, 1.0f);
}

static ShipInput readKeyboard(GLFWwindow* window) {
    ShipInput input;
    input.thrust = glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS;
    input.turnLeft = glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS;
    input.turnRight = glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS;
    return input;
}

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

        Mesh ringMesh(Mesh::makeCircle(1.0f, 24));

        Mesh rectMesh(Mesh::makeRect());

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

        World world(makeTestLevel());

        // Fixed timestep: real time piles up in the accumulator, and the world advances
        // in exact SIM_DT steps however fast or slow the screen refreshes.
        double lastTime = glfwGetTime();
        double accumulator = 0.0;
        while(!glfwWindowShouldClose(window)) {
            double now = glfwGetTime();
            accumulator += std::min(now - lastTime, 0.25);   // cap, so a long freeze doesn't cause hundreds of catch-up steps
            lastTime = now;

            ShipInput input = readKeyboard(window);
            while (accumulator >= constants::SIM_DT) {
                StepResult result = world.step(input);
                if (result == StepResult::ReachedGoal) {
                    std::cout << "Level complete! (" << world.getStepCount() << " steps)\n";
                }
                if (result != StepResult::Running) {
                    world.reset();
                }
                accumulator -= constants::SIM_DT;
            }

            int w, h;
            glfwGetFramebufferSize(window, &w, &h);
            glViewport(0,0,w,h);
            glm::mat4 projection = fitWorldProjection(w, h);

            const Level& level = world.getLevel();

            glClear(GL_COLOR_BUFFER_BIT);
            shader.use();

            shader.setVec4("uColor", constants::COLOR_GRAVITY);
            for (const Obstacle& o : level.obstacles) {
                if (o.density > 0) {
                    glm::mat4 ring = glm::scale(o.modelMatrix(), glm::vec3(constants::GRAVITY_RANGE, constants::GRAVITY_RANGE, 1.0f));
                    shader.setMat4("uTransform", projection * ring);
                    ringMesh.draw(GL_LINE_LOOP);
                }
            }

            shader.setVec4("uColor", constants::COLOR_OBSTACLE);
            for (const Obstacle& o : level.obstacles) {
                shader.setMat4("uTransform", projection * o.modelMatrix());
                circleMesh.draw(GL_LINE_LOOP);
            }

            shader.setVec4("uColor", constants::COLOR_GOAL);
            shader.setMat4("uTransform", projection * level.goal.modelMatrix());
            rectMesh.draw(GL_LINE_LOOP);

            // World boundary, so the edges that kill you are visible
            glm::mat4 bounds = glm::scale(glm::mat4(1.0f), glm::vec3(constants::WORLD_HALF_WIDTH, constants::WORLD_HALF_HEIGHT, 1.0f));
            shader.setVec4("uColor", constants::COLOR_BOUNDS);
            shader.setMat4("uTransform", projection * bounds);
            rectMesh.draw(GL_LINE_LOOP);

            shader.setVec4("uColor", constants::COLOR_SHIP);
            shader.setMat4("uTransform", projection * world.getShip().modelMatrix());
            shipMesh.draw(GL_LINE_LOOP);

            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }
    glfwTerminate();
    return 0;
}
