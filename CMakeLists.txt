cmake_minimum_required(VERSION 3.10)
project(Raycaster)

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# You can define DEBUG_MODE based on your build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG_MODE=1)
else()
    add_compile_definitions(DEBUG_MODE=0)
endif()

find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(Python3_LIBRARY_RELEASE "C:/Libraries/vcpkg/installed/x64-windows/lib/python312.lib"
    CACHE FILEPATH "Path to Python3 release import library (optional)")
set(Python3_LIBRARY_DEBUG "C:/Libraries/vcpkg/installed/x64-windows/debug/lib/python312_d.lib"
    CACHE FILEPATH "Path to Python3 debug import library (optional)")

set(CPP_DIR "${CMAKE_SOURCE_DIR}/cpp_src")
set(CPP_SOURCE_DIR "${CPP_DIR}/src")
set(CPP_INCLUDE_DIR "${CPP_DIR}/include")

include_directories(${CPP_INCLUDE_DIR})

# Add all other files 
file(GLOB_RECURSE CPP_FILES "cpp_src/src/*.cpp")

list(APPEND CPP_FILES "${CPP_DIR}/main.cpp") # Add the main file
add_executable(main ${CPP_FILES}) # Create the main executable

# ---- Libraries ----
target_link_libraries(main PRIVATE
    OpenGL::GL
    glfw
    glad::glad
    glm::glm
    imgui::imgui
    pybind11::embed
)

set(STB_INCLUDE_DIR "${CPP_INCLUDE_DIR}/stb")
target_include_directories(main PRIVATE "${STB_INCLUDE_DIR}")

set(IMGUI_SOURCE_DIR "${CPP_SOURCE_DIR}/imgui")
set(IMGUI_INCLUDE_DIR "${CPP_INCLUDE_DIR}/imgui")
file(GLOB_RECURSE IMGUI_FILES "${IMGUI_SOURCE_DIR}/*.cpp")
target_sources(main PRIVATE "${IMGUI_FILES}")
target_include_directories(main PRIVATE "${IMGUI_INCLUDE_DIR}")

set(SHADER_DIR "${CPP_SOURCE_DIR}/shaders")
set(SHADER_PATH "$<TARGET_FILE_DIR:main>/shaders")
set(TEXTURE_DIR "${CPP_SOURCE_DIR}/textures")
set(TEXTURE_PATH "$<TARGET_FILE_DIR:main>/textures")
set(MODEL_DIR "${CPP_SOURCE_DIR}/models")
set(MODEL_PATH "$<TARGET_FILE_DIR:main>/models")
add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${SHADER_DIR} ${SHADER_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${TEXTURE_DIR} ${TEXTURE_PATH}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${MODEL_DIR} ${MODEL_PATH}
)

set_target_properties(main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_custom_command(TARGET main POST_BUILD
    COMMAND ${Python3_EXECUTABLE} -m venv "$<TARGET_FILE_DIR:main>/.venv"
    COMMAND "$<TARGET_FILE_DIR:main>/.venv/Scripts/python.exe" -m pip install --upgrade pip
    COMMAND "$<TARGET_FILE_DIR:main>/.venv/Scripts/python.exe" -m pip install -r "${PY_DIR}/requirements.txt"
)

# Python link test
set(PY_TEST_EXE_FILE "${CPP_DIR}/py_link_test.cpp") # Add the python main file
add_executable(py_link_test ${CPP_DIR}/${PY_TEST_EXE_FILE})

target_link_libraries(py_link_test PRIVATE
    pybind11::embed
)

set_target_properties(py_link_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test/bin"
)

add_custom_command(TARGET py_link_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PY_DIR} "$<TARGET_FILE_DIR:py_link_test>/py_src"
    COMMAND ${Python3_EXECUTABLE} -m venv "$<TARGET_FILE_DIR:py_link_test>/.venv"
    COMMAND "$<TARGET_FILE_DIR:py_link_test>/.venv/Scripts/python.exe" -m pip install -r "${PY_DIR}/requirements.txt"
)