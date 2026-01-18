cmake_minimum_required(VERSION 3.10)
project(Raycaster)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# point to venv python (fix missing slash). Use the variable name FindPython expects and cache it.
set(Python_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/Scripts/python.exe"
    CACHE FILEPATH "Path to the Python executable in the project venv")

# ---- Find packages ----
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

# ---- Include directories ----
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp_src/include
)

# ---- Source files ----
file(GLOB_RECURSE CPP_SRC_FILES
    "cpp_src/src/*.cpp" # all cpp files in src folder
)

# exclude the test runner from the main executable sources (tester.cpp has its own target)
list(REMOVE_ITEM CPP_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/cpp_src/src/tester.cpp")

list(APPEND CPP_SRC_FILES "cpp_src/main.cpp") # include main.cpp from cpp_src

# ---- Main Executable ----
add_executable(main "${CPP_SRC_FILES}")

# ---- Libraries ----
target_link_libraries(main PRIVATE
    OpenGL::GL
    glfw
    glad::glad
    glm::glm
    imgui::imgui
)

# ---- STB ----
set(STB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cpp_src/include/stb")
if(EXISTS "${STB_DIR}/stb_image.h")
    target_include_directories(main PRIVATE "${STB_DIR}")
else()
    # try to find stb in system/vcpkg includes
    find_path(STB_INCLUDE_DIRS
        NAMES stb_image.h
        PATHS "${CMAKE_CURRENT_SOURCE_DIR}" ENV
        PATH NO_DEFAULT_PATH)
    if(STB_INCLUDE_DIRS)
        target_include_directories(main PRIVATE "${STB_INCLUDE_DIRS}")
    endif()
endif()

# ---- ImGui ----
# Add ImGui headers and sources from your project directories
set(IMGUI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cpp_src/include/imgui")
set(IMGUI_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cpp_src/src/imgui")
set(IMGUI_SOURCES
    "${IMGUI_SRC_DIR}/imgui.cpp"
    "${IMGUI_SRC_DIR}/imgui_draw.cpp"
    "${IMGUI_SRC_DIR}/imgui_tables.cpp"
    "${IMGUI_SRC_DIR}/imgui_widgets.cpp"
    "${IMGUI_SRC_DIR}/imgui_impl_glfw.cpp"
    "${IMGUI_SRC_DIR}/imgui_impl_opengl3.cpp"
)
# Add ImGui sources to your executable
target_sources(main PRIVATE "${IMGUI_SOURCES}")
# Add ImGui include directory
target_include_directories(main PRIVATE "${IMGUI_INCLUDE_DIR}")

# ---- Resources ----
set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cpp_src/src/shaders")
set(TEXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cpp_src/src/textures")
set(MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cpp_src/src/models")
# Copy shaders, textures and models to the output folder after build
add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${SHADER_DIR}" "$<TARGET_FILE_DIR:main>/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${TEXTURE_DIR}" "$<TARGET_FILE_DIR:main>/textures"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${MODEL_DIR}" "$<TARGET_FILE_DIR:main>/models"
)
set_target_properties(main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# # ---- Python test runner executable (optional) ----
# Only add this target if the tester source exists at the expected location.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp_src/tester.cpp")
    add_executable(python_test_runner cpp_src/tester.cpp)
    target_link_libraries(python_test_runner PRIVATE pybind11::embed)
    set_target_properties(python_test_runner PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test_bin"
    )
else()
    message(STATUS "python_test_runner: cpp_src/tester.cpp not found; skipping target")
endif()

# remove or comment out this dependency so main does not try to recompile/link tester into itself
# add_dependencies(main python_test_runner)

# ---- Optional: Debugging definitions ----
# You can define DEBUG_MODE based on your build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG_MODE=1)
else()
    add_compile_definitions(DEBUG_MODE=0)
endif()

# ---- Optional: support embedding Python. Disabled by default; enable with -DUSE_EMBEDDED_PYTHON=ON ----
option(USE_EMBEDDED_PYTHON "Enable embedding Python into the executable" OFF)

# Allow developers to override explicit Python import libraries when needed (cache as FILEPATH)
# Typical defaults point to vcpkg-installed Python libs (adjust per-machine as needed)
set(Python3_LIBRARY_RELEASE "C:/Libraries/vcpkg/installed/x64-windows/lib/python312.lib"
    CACHE FILEPATH "Path to Python3 release import library (optional)")
set(Python3_LIBRARY_DEBUG "C:/Libraries/vcpkg/installed/x64-windows/debug/lib/python312_d.lib"
    CACHE FILEPATH "Path to Python3 debug import library (optional)")

if(USE_EMBEDDED_PYTHON)
    set(SETUP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cpp_src/src/setup")
    add_custom_command(TARGET main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${SETUP_DIR}" "$<TARGET_FILE_DIR:main>/setup"
        COMMAND "$<TARGET_FILE_DIR:main>/setup/setup_venv.bat"
    )
    
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    if(Python3_FOUND)
        message(STATUS "Embedding Python: ${Python3_VERSION}")
        message(STATUS "Python3_EXECUTABLE=${Python3_EXECUTABLE}")
        message(STATUS "Python3_LIBRARY_RELEASE=${Python3_LIBRARY_RELEASE}")
        message(STATUS "Python3_LIBRARY_DEBUG=${Python3_LIBRARY_DEBUG}")

        target_include_directories(main PRIVATE "${Python3_INCLUDE_DIRS}")
        # Add both debug and release library directories to the link search path.
        if(DEFINED Python3_LIBRARY_RELEASE)
            get_filename_component(_PY3_LIB_DIR_RELEASE "${Python3_LIBRARY_RELEASE}" DIRECTORY)
            target_link_directories(main PRIVATE "${_PY3_LIB_DIR_RELEASE}")
        endif()
        if(DEFINED Python3_LIBRARY_DEBUG)
            get_filename_component(_PY3_LIB_DIR_DEBUG "${Python3_LIBRARY_DEBUG}" DIRECTORY)
            target_link_directories(main PRIVATE "${_PY3_LIB_DIR_DEBUG}")
        endif()

        # Some Python distributions / wrappers inject an unqualified library name (e.g. pragma comment(lib, "python312.lib"))
        # Ensure the common vcpkg library directory is in the link search path as a fallback so the linker can resolve those unqualified names.
        if(EXISTS "C:/Libraries/vcpkg/installed/x64-windows/lib")
            target_link_directories(main PRIVATE "C:/Libraries/vcpkg/installed/x64-windows/lib")
        endif()
        # Prefer explicit per-configuration library paths when available to avoid missing-linker-search-path issues
        if(Python3_LIBRARY_RELEASE AND Python3_LIBRARY_DEBUG)
            target_link_libraries(main PRIVATE
                "$<$<CONFIG:Debug>:${Python3_LIBRARY_DEBUG}>"
                "$<$<CONFIG:Release>:${Python3_LIBRARY_RELEASE}>"
            )
        elseif(Python3_LIBRARY)
            target_link_libraries(main PRIVATE "${Python3_LIBRARY}")
        else()
            target_link_libraries(main PRIVATE "${Python3_LIBRARIES}")
        endif()
        target_compile_definitions(main PRIVATE USE_EMBEDDED_PYTHON=1)
    else()
        target_compile_definitions(main PRIVATE USE_EMBEDDED_PYTHON=0)
    endif()
endif()

# ---- Optional: Copy py_src directory after build (ON, OFF) ----
option(COPY_PY_SRC "Copy py_src directory after build" ON)
if(COPY_PY_SRC)
    add_custom_command(TARGET main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/py_src" "$<TARGET_FILE_DIR:main>/py_src"
    )
endif()