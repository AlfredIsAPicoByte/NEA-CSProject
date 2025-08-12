cmake_minimum_required(VERSION 3.10)
project(Raycaster)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_path(STB_INCLUDE_DIRS "stb_image.h")

add_executable(main cpp_src/main.cpp
    cpp_src/shaderClass.cpp
    cpp_src/textureClass.cpp
    cpp_src/VBO.cpp
    cpp_src/VAO.cpp
    cpp_src/EBO.cpp
    cpp_src/stb_image.cpp
)
target_include_directories(main PRIVATE cpp_src/include)

target_link_libraries(main PRIVATE
    OpenGL::GL
    glfw
    glad::glad
    glm::glm
)
target_include_directories(main PRIVATE ${STB_INCLUDE_DIRS})

set(SHADER_DIR "${CMAKE_SOURCE_DIR}/cpp_src/shaders")
set(TEXTURE_DIR "${CMAKE_SOURCE_DIR}/cpp_src/textures")

add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${SHADER_DIR} $<TARGET_FILE_DIR:main>/shaders
    
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${TEXTURE_DIR} $<TARGET_FILE_DIR:main>/textures
)
