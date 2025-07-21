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
| OpenGL       | 4.6      | via GPU drivers |

## 🔷 Environment Variables

- `PATH` includes:
  - C:\msys64\mingw64\bin
  - C:\Program Files\CMake\bin
  - C:\Python311\Scripts

## 🔷 Build Commands

```bash
cmake -S . -B build
cmake --build build