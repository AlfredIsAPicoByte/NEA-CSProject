# Project Requirements (updated)

## System
- Windows 10/11 (x64)
- PowerShell 5.1+ or PowerShell Core 7+

## Tools (recommended)
- CMake >= 3.20 (3.28+ recommended)
- Visual Studio 2022 or newer (MSVC) or MSYS2 + MinGW for GCC
- vcpkg for dependency management (recommended)
- Python 3.11 or 3.12 (match installed interpreter with development files)
- pybind11 (via vcpkg or pip; prefer vcpkg for C++ integration)
- GLFW (via vcpkg or built)
- glad (generated for your OpenGL version; include KHR_debug if using debug output)
- glm (via vcpkg)
- stb_image (header-only; included in repo)

## Python / pybind11 notes
- If embedding or using pybind11 headers, you need Python development files (include dir with Python.h and pythonXY.lib for MSVC).
- For CMake, prefer FindPython / Python3::Python. If using a virtualenv, set:
  - -DPython3_ROOT_DIR="C:/path/to/venv"
  - or set Python3_ROOT to the interpreter that contains dev files.
- Install runtime deps for tester:
  pip install -r py_src/requirements.txt
  (create this file if needed; at minimum include numpy)

## Environment variables
- PATH should include:
  - C:\Program Files\CMake\bin
  - C:\Libraries\vcpkg\installed\x64-windows\bin (if needed)
  - python Scripts path if using virtualenv
- Ensure vcpkg toolchain path is available when configuring with CMake:
  -DCMAKE_TOOLCHAIN_FILE=C:/Libraries/vcpkg/scripts/buildsystems/vcpkg.cmake

## OpenGL
- GPU drivers must support the target OpenGL version used by glad.
- For KHR_debug and modern features use OpenGL 4.3+, recommend 4.6.

## Build (examples)
PowerShell (recommended):
````powershell
# configure
cmake -S . -B build\debug `
  -DCMAKE_BUILD_TYPE=Debug `
  -DCMAKE_TOOLCHAIN_FILE="C:/Libraries/vcpkg/scripts/buildsystems/vcpkg.cmake" `
  -DUSE_EMBEDDED_PYTHON=OFF

# build main
cmake --build build\debug --config Debug --target main

# build python test runner (if present)
cmake --build build\debug --config Debug --target python_test_runner
````