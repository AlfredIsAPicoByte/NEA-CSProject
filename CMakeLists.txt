cmake_minimum_required(VERSION 3.10)
project(Raycaster)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenGL REQUIRED)
set(glfw3_DIR "C:/Libraries/vcpkg/packages/glfw3_x64-windows/share/glfw3")
find_package(glfw3 REQUIRED)

add_executable(main main.cpp)

target_link_libraries(main PRIVATE OpenGL::GL glfw)