cmake_minimum_required(VERSION 3.10)
project(Raycaster)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Find packages ----
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# ---- Include directories ----
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp_src/include
)
target_include_directories(main PRIVATE /path/to/pybind11/include)

# ---- Source files ----
file(GLOB_RECURSE CPP_SRC_FILES
    cpp_src/src/*.cpp                           # all cpp files in src folder
)
list(APPEND CPP_SRC_FILES cpp_src/main.cpp)     # include main.cpp from cpp_src

# ---- Executable ----
add_executable(main ${CPP_SRC_FILES})

# ---- Libraries ----
target_link_libraries(main PRIVATE
    OpenGL::GL
    glfw
    glad::glad
    glm::glm
    pybind11::embed
    ${Python_LIBRARIES}
)
target_include_directories(main PRIVATE ${Python_INCLUDE_DIRS})

# ---- stb_image ----
find_path(STB_INCLUDE_DIRS "stb_image.h")
if(STB_INCLUDE_DIRS)
    target_include_directories(main PRIVATE ${STB_INCLUDE_DIRS})
endif()

# ---- Resources ----
set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cpp_src/src/shaders")
set(TEXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cpp_src/src/textures")

# Copy shaders and textures to the output folder
add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${SHADER_DIR} $<TARGET_FILE_DIR:main>/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${TEXTURE_DIR} $<TARGET_FILE_DIR:main>/textures
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/py_src $<TARGET_FILE_DIR:main>/py_src
)

# ---- Optional: support embedding Python. Disabled by default; enable with -DUSE_EMBEDDED_PYTHON=ON ----
option(USE_EMBEDDED_PYTHON "Enable embedding Python into the executable" OFF)
if(USE_EMBEDDED_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    if(Python3_FOUND)
        message(STATUS "Embedding Python: ${Python3_VERSION}")
        target_include_directories(main PRIVATE ${Python3_INCLUDE_DIRS})
        target_link_libraries(main PRIVATE ${Python3_LIBRARIES})
        target_compile_definitions(main PRIVATE USE_EMBEDDED_PYTHON=1)
    endif()
endif()

# ---- Optional: Output folders ----
set_target_properties(main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ---- Optional: Debugging definitions ----
# You can define DEBUG_MODE based on your build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG_MODE=1)
else()
    add_compile_definitions(DEBUG_MODE=0)
endif()