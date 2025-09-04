cmake_minimum_required(VERSION 3.10)
project(Raycaster)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Explicitly point to Python installation
find_package(Python COMPONENTS Development REQUIRED)

# ---- Find packages ----
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

# ---- Include directories ----
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp_src/include
)

include_directories(main PRIVATE ${STB_INCLUDE_DIRS} ${Python_INCLUDE_DIRS})

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
    ${Python_LIBRARIES}
)

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
)

# ---- Optional: Output folders ----
set_target_properties(main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
