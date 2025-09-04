# Project Requirements

## 🔷 System

- Windows 11 (Build 22000+)
- PowerShell Core 7.3+ or Windows PowerShell 5.1+

## 🔷 Tools

| Tool         | Version  | Install Method |
|--------------|----------|----------------|
| CMake        | 3.28.1   | https://cmake.org/download/ |
| g++ (MinGW)  | 12.2.0   | MSYS2 |
| Python       | 3.11.4   | https://python.org |
| Pybind11     | 2.11.1   | pip install pybind11 |
| GLFW         | 3.3.8    | built from source |
| OpenGL       | 4.6\*    | via GPU drivers |
| glad         | 0.1.36+  | [glad.dav1d.de](https://glad.dav1d.de/) (generated for OpenGL 4.6 Core, with KHR_debug) |
| glm          | 0.9.9+   | vcpkg or built from source |
| stb_image    | latest   | header-only, included in project |

## 🔷 Environment Variables

- `PATH` includes:
  - C:\msys64\mingw64\bin
  - C:\Program Files\CMake\bin
  - C:\Python311\Scripts
- `Library` points to your external libraries folder (e.g., `C:\Libraries`)

## 🔷 OpenGL Requirements

- **GPU and drivers must support OpenGL 4.3 or higher** for debugging features (KHR_debug).
- If your GPU only supports OpenGL 3.3, advanced debugging will not be available.

## 🔷 Build Commands

```bash
cmake -S . -B Build
cmake --build Build
```

\* Ensure your glad loader and CMake configuration match the OpenGL version supported by your GPU