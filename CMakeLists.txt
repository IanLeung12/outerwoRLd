cmake_minimum_required(VERSION 3.20)
project(outerworld LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The simulation: no window, no OpenGL. Shared by the game and the Python DLL.
add_library(outerworld_core STATIC src/Ship.cpp src/World.cpp src/LevelGen.cpp src/DistanceField.cpp)
target_include_directories(outerworld_core PUBLIC include src)
# Training runs the simulation millions of times, so it's always optimized, even in a debug build
target_compile_options(outerworld_core PRIVATE -O2)

# The game / viewer
add_executable(outerworld src/main.cpp src/glad.c src/Shader.cpp src/Mesh.cpp)

target_link_libraries(outerworld PRIVATE
    outerworld_core
    ${CMAKE_SOURCE_DIR}/lib/libglfw3.a
    opengl32
    gdi32
)

# The DLL that Python loads (python/outerworld_env.py). Linked statically so it
# doesn't need MinGW's runtime DLLs next to it.
add_library(outerworld_py SHARED src/capi.cpp)
target_link_libraries(outerworld_py PRIVATE outerworld_core)
target_link_options(outerworld_py PRIVATE -static)
set_target_properties(outerworld_py PROPERTIES PREFIX "")
