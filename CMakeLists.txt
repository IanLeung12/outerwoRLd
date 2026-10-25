cmake_minimum_required(VERSION 3.20)
project(outerworld LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(outerworld src/main.cpp src/glad.c src/Shader.cpp src/Mesh.cpp src/Ship.cpp)

target_include_directories(outerworld PRIVATE include)

target_link_libraries(outerworld PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/libglfw3.a
    opengl32
    gdi32
)