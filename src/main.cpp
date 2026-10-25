
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "constants.h"
#include "Shader.h"
#include "Mesh.h"
#include "LevelGen.h"
#include "World.h"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
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

// Which level to play: the hand-made test level, or a generated one. Written in replays as
// "test" or "generated <seed> <difficulty> [<blocker chance>]".
struct LevelChoice {
    bool generated = false;
    uint32_t seed = 0;
    float difficulty = 0.5f;
    float blockerChance = 0.0f;
};

static Level buildLevel(const LevelChoice& choice) {
    if (!choice.generated) return makeTestLevel();
    GenParams params = paramsForDifficulty(choice.difficulty);
    params.blockerChance = choice.blockerChance;
    return generateLevel(choice.seed, params);
}

static std::string describe(const LevelChoice& choice) {
    if (!choice.generated) return "test";
    std::string text = "generated " + std::to_string(choice.seed) + " " + std::to_string(choice.difficulty);
    if (choice.blockerChance > 0.0f) text += " " + std::to_string(choice.blockerChance);
    return text;
}

static bool parseLevel(std::istream& in, LevelChoice& choice) {
    std::string name;
    if (!(in >> name)) return false;
    if (name == "test") {
        choice.generated = false;
        return true;
    }
    if (name == "generated" && in >> choice.seed >> choice.difficulty) {
        choice.generated = true;
        if (!(in >> choice.blockerChance)) choice.blockerChance = 0.0f;   // optional
        return true;
    }
    std::cerr << "Unknown level '" << name << "'\n";
    return false;
}

// A recorded episode, saved by OuterworldEnv.save_replay() in Python. The simulation is
// deterministic, so the level plus the list of actions recreates the episode exactly.
struct Replay {
    LevelChoice level;
    std::vector<int> actions;
};

static bool loadReplay(const char* path, Replay& replay) {
    std::ifstream file(path);
    std::string firstLine, keyword;
    std::getline(file, firstLine);
    std::istringstream header(firstLine);
    if (!(header >> keyword) || keyword != "level" || !parseLevel(header, replay.level)) {
        std::cerr << "Couldn't read replay " << path << "\n";
        return false;
    }
    int action;
    while (file >> action) {
        if (action < 0 || action >= NUM_ACTIONS) {
            std::cerr << "Invalid action " << action << " in replay\n";
            return false;
        }
        replay.actions.push_back(action);
    }
    if (replay.actions.empty()) {
        std::cerr << "Replay " << path << " has no actions\n";
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    // Usage: outerworld                              play the test level (W thrust, A/D turn, Esc quit)
    //        outerworld --seed <n> [--difficulty <d>]  play a generated level (default difficulty 0.5)
    //                   [--blocker <chance>]           ... with a planet in the way (chance 0..1)
    //        outerworld --replay <file>              watch a recorded episode, on a loop
    // While playing, N / P switch to the next / previous generated level.
    bool replaying = false;
    Replay replay;
    LevelChoice choice;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string flag = argv[i];
        if (flag == "--replay") {
            replaying = true;
            if (!loadReplay(argv[i + 1], replay)) return 1;
            choice = replay.level;
        } else if (flag == "--seed") {
            choice.generated = true;
            choice.seed = (uint32_t)std::stoul(argv[i + 1]);
        } else if (flag == "--difficulty") {
            choice.difficulty = std::stof(argv[i + 1]);
        } else if (flag == "--blocker") {
            choice.blockerChance = std::stof(argv[i + 1]);
        } else {
            std::cerr << "Unknown option " << flag << "\n";
            return 1;
        }
    }

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

        World world(buildLevel(choice));
        size_t replayIndex = 0;
        if (replaying) {
            std::cout << "Replaying " << replay.actions.size() << " steps on level '" << describe(choice) << "'\n";
        } else {
            std::cout << "Level '" << describe(choice) << "' (N / P for next / previous generated level)\n";
        }
        bool nWasDown = false, pWasDown = false;

        // Fixed timestep: real time piles up in the accumulator, and the world advances
        // in exact SIM_DT steps however fast or slow the screen refreshes.
        double lastTime = glfwGetTime();
        double accumulator = 0.0;
        while(!glfwWindowShouldClose(window)) {
            double now = glfwGetTime();
            accumulator += std::min(now - lastTime, 0.25);   // cap, so a long freeze doesn't cause hundreds of catch-up steps
            lastTime = now;

            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                glfwSetWindowShouldClose(window, true);
            }

            // N / P: step through generated levels (only on the key press, not while held)
            bool nDown = glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS;
            bool pDown = glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS;
            if (!replaying && ((nDown && !nWasDown) || (pDown && !pWasDown))) {
                if (choice.generated) {
                    choice.seed += nDown ? 1 : -1;
                } else {
                    choice.generated = true;   // from the test level, start at seed 0
                }
                world.load(buildLevel(choice));
                std::cout << "Level '" << describe(choice) << "'\n";
            }
            nWasDown = nDown;
            pWasDown = pDown;

            ShipInput keyboard = readKeyboard(window);
            while (accumulator >= constants::SIM_DT) {
                bool replayFinished = replaying && replayIndex >= replay.actions.size();
                StepResult result = StepResult::Running;
                if (!replayFinished) {
                    ShipInput input = replaying ? decodeAction(replay.actions[replayIndex++]) : keyboard;
                    result = world.step(input);
                }

                if (result == StepResult::ReachedGoal) {
                    std::cout << "Level complete! (" << world.getStepCount() << " steps)\n";
                } else if (result == StepResult::Crashed && replaying) {
                    std::cout << "Crashed after " << world.getStepCount() << " steps\n";
                } else if (replayFinished) {
                    std::cout << "Timed out after " << world.getStepCount() << " steps\n";
                }
                if (result != StepResult::Running || replayFinished) {
                    world.reset();
                    replayIndex = 0;   // replays loop
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
